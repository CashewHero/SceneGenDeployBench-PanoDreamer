from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from runner_wrapper.files import publish_directory, publish_file

logger = logging.getLogger("runner_wrapper.server")


def configure_logging() -> None:
    runner_level_name = os.getenv("RUNNER_LOG_LEVEL", "INFO").upper()
    third_party_level_name = os.getenv("THIRD_PARTY_LOG_LEVEL", "WARNING").upper()
    runner_level = getattr(logging, runner_level_name, logging.INFO)
    third_party_level = getattr(
        logging,
        third_party_level_name,
        logging.WARNING,
    )
    logging.basicConfig(
        level=third_party_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("runner_wrapper").setLevel(runner_level)


def event_message(event: str, **fields: object) -> str:
    return json.dumps({"event": event, **fields}, sort_keys=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_run_job_handler(target: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    module_name, separator, attribute_name = target.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "RUNNER_ADAPTER must use the format 'module.path:function_name', "
            f"received {target!r}"
        )

    module = import_module(module_name)
    handler = getattr(module, attribute_name)
    if not callable(handler):
        raise TypeError(f"configured adapter target is not callable: {target}")
    return handler


@dataclass(frozen=True)
class RunnerSettings:
    port: int
    runner_name: str
    runner_type: str
    runner_version: str
    contract_version: int
    idle_timeout_seconds: int
    startup_timeout_seconds: float
    adapter_target: str

    @classmethod
    def from_env(cls) -> "RunnerSettings":
        return cls(
            port=int(os.getenv("RUNNER_PORT", "58090")),
            runner_name=os.getenv("RUNNER_NAME", "runner"),
            runner_type=os.getenv("RUNNER_TYPE", "generator"),
            runner_version=os.getenv("RUNNER_VERSION", "0.1.0"),
            contract_version=int(os.getenv("RUNNER_CONTRACT_VERSION", "1")),
            idle_timeout_seconds=int(os.getenv("RUNNER_IDLE_TIMEOUT_SECONDS", "900")),
            startup_timeout_seconds=float(os.getenv("RUNNER_STARTUP_TIMEOUT_SECONDS", "60")),
            adapter_target=os.getenv("RUNNER_ADAPTER", "runner_wrapper.adapter:run_job"),
        )


def make_status_payload(runner: "Runner", accepted: bool | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "online": True,
        "runner_name": runner.settings.runner_name,
        "runner_type": runner.settings.runner_type,
        "runner_version": runner.settings.runner_version,
        "contract_version": runner.settings.contract_version,
        "batch_id": runner.batch_id,
        "state": runner.state,
        "current_job_id": runner.current_job_id,
        "updated_at": runner.updated_at,
        "result": runner.result if runner.state in ("finished", "failed") else None,
    }
    if accepted is not None:
        payload["accepted"] = accepted
    return payload


def build_failure_result(exc: Exception) -> dict[str, Any]:
    completed_at = utc_now()
    return {
        "status": "failed",
        "started_at": completed_at,
        "completed_at": completed_at,
        "metrics": [],
        "artifacts": [],
        "failure": {
            "code": "RUNNER_INTERNAL_ERROR",
            "message": str(exc),
            "retryable": False,
            "stage": "runner",
            "traceback": traceback.format_exc(),
        },
    }


def request_job(job_request: dict[str, Any]) -> dict[str, Any]:
    job = job_request.get("job")
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    return job


def request_runtime(job_request: dict[str, Any]) -> dict[str, Any]:
    runtime = job_request.get("runtime")
    return runtime if isinstance(runtime, dict) else {}


def job_timeout_seconds(job_request: dict[str, Any]) -> float:
    raw_timeout = request_job(job_request).get("timeout_seconds")
    if raw_timeout in (None, ""):
        raise ValueError("job.timeout_seconds is required")
    timeout_seconds = float(raw_timeout)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("job.timeout_seconds must be a finite number greater than 0")
    return timeout_seconds


def result_publish_paths(result: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    output_files = result.get("output_files")
    if isinstance(output_files, dict):
        for sample_files in output_files.values():
            if not isinstance(sample_files, dict):
                continue
            for path in sample_files.values():
                if isinstance(path, str) and path.strip():
                    paths.add(path)

    artifacts = result.get("artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            path = artifact.get("path")
            if isinstance(path, str) and path.strip():
                paths.add(path)
    return paths


def publish_result_files(
    result: dict[str, Any],
    workspace_dir: str | Path,
    output_dir: str | Path,
) -> None:
    workspace_path = Path(workspace_dir).resolve()
    output_path = Path(output_dir)
    for relative_path in sorted(result_publish_paths(result)):
        path = Path(relative_path)
        if path.is_absolute() or path == Path(".") or ".." in path.parts:
            raise ValueError(f"published path must be a relative file or directory: {relative_path}")
        source_path = (workspace_path / path).resolve()
        try:
            source_path.relative_to(workspace_path)
        except ValueError as exc:
            raise ValueError(f"published path escapes the workspace: {relative_path}") from exc
        destination_path = output_path / path
        if source_path.is_file():
            publish_file(source_path, destination_path)
        elif source_path.is_dir():
            publish_directory(source_path, destination_path, dirs_exist_ok=True)
        else:
            raise FileNotFoundError(f"published path not found in workspace: {relative_path}")


def build_timeout_result(job_id: object, timeout_seconds: float, kill_after_seconds: float) -> dict[str, Any]:
    completed_at = utc_now()
    return {
        "status": "failed",
        "started_at": completed_at,
        "completed_at": completed_at,
        "metrics": [],
        "artifacts": [],
        "failure": {
            "code": "RUNNER_JOB_TIMEOUT",
            "message": f"job {job_id} exceeded timeout_seconds + 60 seconds",
            "retryable": True,
            "stage": "runner",
            "timeout_seconds": timeout_seconds,
            "kill_after_seconds": kill_after_seconds,
        },
    }


def build_process_failure_result(job_id: object, exit_code: int | None) -> dict[str, Any]:
    completed_at = utc_now()
    return {
        "status": "failed",
        "started_at": completed_at,
        "completed_at": completed_at,
        "metrics": [],
        "artifacts": [],
        "failure": {
            "code": "RUNNER_PROCESS_EXITED",
            "message": f"job {job_id} process exited without returning a result",
            "retryable": True,
            "stage": "runner",
            "exit_code": exit_code,
        },
    }


def run_job_child(
    run_job_handler: Callable[[dict[str, Any]], dict[str, Any]],
    job_request: dict[str, Any],
    result_queue: multiprocessing.Queue,
) -> None:
    configure_logging()
    try:
        runtime = request_runtime(job_request)
        output_dir = runtime.get("output_dir")
        if isinstance(output_dir, str) and output_dir.strip():
            with tempfile.TemporaryDirectory(prefix="runner-output-") as temp_dir:
                staging_dir = Path(temp_dir) / "output"
                staging_dir.mkdir()
                staged_request = dict(job_request)
                adapter_runtime = dict(runtime)
                adapter_runtime.pop("output_dir", None)
                adapter_runtime["workspace_dir"] = str(staging_dir)
                staged_request["runtime"] = adapter_runtime
                result = run_job_handler(staged_request)
                publish_result_files(result, staging_dir, output_dir)
        else:
            result = run_job_handler(job_request)
        result_queue.put({"ok": True, "result": result})
    except Exception as exc:
        result_queue.put({"ok": False, "result": build_failure_result(exc)})


def validate_job_request(job_request: object) -> str | None:
    if not isinstance(job_request, dict):
        return "request body must be a JSON object"
    try:
        job = request_job(job_request)
        if not str(job.get("batch_id") or "").strip():
            return "job.batch_id is required"
        if not isinstance(job.get("parameters", {}), dict):
            return "job.parameters must be an object"
        job_timeout_seconds(job_request)
    except (TypeError, ValueError) as exc:
        return str(exc)
    return None


@dataclass
class Runner:
    settings: RunnerSettings
    run_job_handler: Callable[[dict[str, Any]], dict[str, Any]]
    state: str = "starting"
    batch_id: str | None = None
    current_job_id: str | None = None
    result: dict[str, Any] | None = None
    updated_at: str = field(default_factory=utc_now)
    last_status_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    child_process: multiprocessing.Process | None = field(default=None, init=False, repr=False)

    def set_state(
        self,
        state: str,
        current_job_id: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        previous_state = self.state
        previous_job_id = self.current_job_id
        self.state = state
        self.current_job_id = current_job_id
        self.result = result
        self.updated_at = utc_now()
        if previous_state != state or previous_job_id != current_job_id:
            logger.info(
                event_message(
                    "runner_state_changed",
                    previous_state=previous_state,
                    state=state,
                    job_id=current_job_id,
                )
            )

    def mark_ready(self) -> None:
        with self.lock:
            self.set_state("idle")

    def touch_status(self) -> None:
        with self.lock:
            self.last_status_at = time.time()
            self.updated_at = utc_now()

    def submit_job(self, job_request: dict[str, Any]) -> tuple[bool, str | None]:
        reject_reason = validate_job_request(job_request)
        if reject_reason is not None:
            logger.warning(event_message("job_rejected", reason=reject_reason))
            return False, reject_reason

        job = request_job(job_request)
        job_id = job.get("job_id")
        batch_id = str(job.get("batch_id") or "").strip()
        with self.lock:
            if self.state not in ("idle", "finished", "failed"):
                reject_reason = f"runner is not available: {self.state}"
                logger.warning(event_message("job_rejected", job_id=job_id, state=self.state, batch_id=batch_id))
                return False, reject_reason
            if self.batch_id and self.batch_id != batch_id:
                reject_reason = "job.batch_id does not match the batch already bound to this runner"
                logger.warning(
                    event_message(
                        "job_rejected",
                        job_id=job_id,
                        state=self.state,
                        batch_id=batch_id,
                        bound_batch_id=self.batch_id,
                        reason="batch_mismatch",
                    )
                )
                return False, reject_reason
            if self.batch_id is None:
                self.batch_id = batch_id
            self.set_state("running", current_job_id=job_id, result=None)
            logger.info(
                event_message(
                    "job_accepted",
                    job_id=job_id,
                    batch_id=batch_id,
                )
            )

        worker = threading.Thread(target=self._run_job_thread, args=(job_request,), daemon=True)
        worker.start()
        return True, None

    def _run_job_thread(self, job_request: dict[str, Any]) -> None:
        job = request_job(job_request)
        job_id = job.get("job_id")
        try:
            timeout_seconds = job_timeout_seconds(job_request)
            kill_after_seconds = timeout_seconds + 60.0
            logger.info(
                event_message(
                    "job_execution_started",
                    job_id=job_id,
                    output_dir=request_runtime(job_request).get("output_dir"),
                )
            )
            result = self._run_job_in_child_process(job_request, timeout_seconds, kill_after_seconds)
            with self.lock:
                if self.state == "shutting_down":
                    logger.info(event_message("job_execution_interrupted", job_id=job_id))
                    return
                final_state = "finished" if result.get("status") == "completed" else "failed"
                self.set_state(final_state, current_job_id=job_id, result=result)
                logger.info(
                    event_message(
                        "job_execution_finished",
                        job_id=job_id,
                        state=final_state,
                        result_status=result.get("status"),
                        artifact_count=len(result.get("artifacts", [])),
                        metric_count=len(result.get("metrics", [])),
                    )
                )
        except Exception as exc:
            failure = build_failure_result(exc)
            with self.lock:
                if self.state == "shutting_down":
                    logger.info(event_message("job_execution_interrupted", job_id=job_id))
                    return
                self.set_state("failed", current_job_id=job_id, result=failure)
            logger.exception(event_message("job_execution_failed", job_id=job_id, error=str(exc)))

    def _run_job_in_child_process(
        self,
        job_request: dict[str, Any],
        timeout_seconds: float,
        kill_after_seconds: float,
    ) -> dict[str, Any]:
        job_id = request_job(job_request).get("job_id")
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=run_job_child,
            args=(self.run_job_handler, job_request, result_queue),
        )
        process.start()
        with self.lock:
            self.child_process = process
        try:
            deadline = time.monotonic() + kill_after_seconds
            payload: dict[str, Any] | None = None
            while payload is None:
                with self.lock:
                    shutdown_requested = self.state == "shutting_down"
                if shutdown_requested:
                    self._stop_child_process(process)
                    return build_process_failure_result(job_id, process.exitcode)

                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    break
                try:
                    candidate = result_queue.get(timeout=min(0.1, remaining_seconds))
                except queue.Empty:
                    if not process.is_alive():
                        break
                    continue
                if not isinstance(candidate, dict):
                    raise TypeError("runner child returned an invalid result envelope")
                payload = candidate

            if payload is None and process.is_alive():
                logger.error(
                    event_message(
                        "runner_job_timeout",
                        job_id=job_id,
                        timeout_seconds=timeout_seconds,
                        kill_after_seconds=kill_after_seconds,
                        child_pid=process.pid,
                    )
                )
                self._stop_child_process(process)
                return build_timeout_result(job_id, timeout_seconds, kill_after_seconds)

            process.join(1)
            if payload is None:
                return build_process_failure_result(job_id, process.exitcode)
            if payload.get("ok"):
                result = payload.get("result")
                if not isinstance(result, dict):
                    raise TypeError("run_job must return a dict")
                return result
            failure_result = payload.get("result")
            if isinstance(failure_result, dict):
                return failure_result
            return build_process_failure_result(job_id, process.exitcode)
        finally:
            with self.lock:
                if self.child_process is process:
                    self.child_process = None
            result_queue.close()
            result_queue.join_thread()

    @staticmethod
    def _stop_child_process(process: multiprocessing.Process) -> None:
        if not process.is_alive():
            return
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)

    def request_shutdown(self) -> bool:
        with self.lock:
            interrupted_job_id = self.current_job_id if self.state == "running" else None
            child_process = self.child_process
            self.set_state("shutting_down")
            logger.info(
                event_message(
                    "shutdown_requested",
                    state=self.state,
                    interrupted_job_id=interrupted_job_id,
                )
            )
        if child_process is not None:
            self._stop_child_process(child_process)
        return True


class RunnerHandler(BaseHTTPRequestHandler):
    server_version = "RunnerWrapper/1.0"

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path != "/status":
            logger.warning(event_message("http_not_found", method="GET", path=self.path))
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        self.server.runner.touch_status()
        with self.server.runner.lock:
            payload = make_status_payload(self.server.runner)
        self._send_json(HTTPStatus.OK, payload)

    def do_POST(self) -> None:
        if self.path == "/run-job":
            self._handle_run_job()
            return
        if self.path == "/shutdown":
            self._handle_shutdown()
            return

        logger.warning(event_message("http_not_found", method="POST", path=self.path))
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _handle_run_job(self) -> None:
        try:
            payload = self._read_json()
        except (json.JSONDecodeError, ValueError):
            logger.warning(event_message("invalid_json", path=self.path))
            self._send_json(HTTPStatus.BAD_REQUEST, {"accepted": False, "error": "invalid json"})
            return
        if not isinstance(payload, dict):
            error = "request body must be a JSON object"
            logger.warning(event_message("invalid_request", path=self.path, error=error))
            self._send_json(HTTPStatus.BAD_REQUEST, {"accepted": False, "error": error})
            return

        accepted, error = self.server.runner.submit_job(payload)
        with self.server.runner.lock:
            response = make_status_payload(self.server.runner, accepted=accepted)
        if error is not None:
            response["error"] = error
        self._send_json(HTTPStatus.OK, response)

    def _handle_shutdown(self) -> None:
        accepted = self.server.runner.request_shutdown()
        with self.server.runner.lock:
            response = make_status_payload(self.server.runner, accepted=accepted)
        self._send_json(HTTPStatus.OK, response)
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format: str, *args: object) -> None:
        return


class RunnerHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[RunnerHandler], runner: Runner):
        super().__init__(server_address, handler_class)
        self.runner = runner


def idle_shutdown_loop(server: RunnerHTTPServer) -> None:
    while True:
        time.sleep(5)
        with server.runner.lock:
            if server.runner.state == "running":
                continue

            idle_for = time.time() - server.runner.last_status_at
            if (
                server.runner.state != "shutting_down"
                and idle_for >= server.runner.settings.idle_timeout_seconds
            ):
                logger.info(
                    event_message(
                        "idle_shutdown_triggered",
                        idle_seconds=round(idle_for, 3),
                    )
                )
                server.runner.set_state("shutting_down")

            if server.runner.state == "shutting_down":
                break

    logger.info(event_message("server_shutdown"))
    server.shutdown()


def start_startup_timeout_watchdog(settings: RunnerSettings) -> threading.Event:
    ready_event = threading.Event()
    if settings.startup_timeout_seconds <= 0:
        raise ValueError("RUNNER_STARTUP_TIMEOUT_SECONDS must be greater than 0")

    def watch_startup() -> None:
        if ready_event.wait(settings.startup_timeout_seconds):
            return
        logger.error(
            event_message(
                "runner_startup_timeout",
                timeout_seconds=settings.startup_timeout_seconds,
                adapter_target=settings.adapter_target,
            )
        )
        os._exit(124)

    threading.Thread(target=watch_startup, daemon=True).start()
    return ready_event


def main() -> None:
    configure_logging()
    settings = RunnerSettings.from_env()
    startup_ready = start_startup_timeout_watchdog(settings)
    run_job_handler = load_run_job_handler(settings.adapter_target)
    logger.info(
        event_message(
            "runner_starting",
            port=settings.port,
            runner_name=settings.runner_name,
            runner_version=settings.runner_version,
            idle_timeout_seconds=settings.idle_timeout_seconds,
            startup_timeout_seconds=settings.startup_timeout_seconds,
            adapter_target=settings.adapter_target,
        )
    )

    runner = Runner(settings=settings, run_job_handler=run_job_handler)
    server = RunnerHTTPServer(("0.0.0.0", settings.port), RunnerHandler, runner)
    runner.mark_ready()
    startup_ready.set()

    shutdown_thread = threading.Thread(target=idle_shutdown_loop, args=(server,), daemon=True)
    shutdown_thread.start()

    try:
        server.serve_forever()
    finally:
        logger.info(event_message("server_closed"))
        server.server_close()


if __name__ == "__main__":
    main()
