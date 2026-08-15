from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path

from runner_wrapper.server import (
    Runner,
    RunnerHandler,
    RunnerHTTPServer,
    RunnerSettings,
    validate_job_request,
)


def completed_job(_: dict) -> dict:
    return {
        "status": "completed",
        "metrics": [],
        "artifacts": [],
        "failure": None,
    }


def slow_job(_: dict) -> dict:
    time.sleep(1)
    return completed_job({})


def large_result_job(_: dict) -> dict:
    result = completed_job({})
    result["payload"] = "x" * (2 * 1024 * 1024)
    return result


def output_job(job_request: dict) -> dict:
    runtime = job_request["runtime"]
    workspace_dir = Path(runtime["workspace_dir"])
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "result.txt").write_text("published\n", encoding="utf-8")
    (workspace_dir / "report.txt").write_text("artifact\n", encoding="utf-8")
    (workspace_dir / "scratch.tmp").write_text("temporary\n", encoding="utf-8")
    result = completed_job({})
    result["output_files"] = {"sample-1": {"text": "result.txt"}}
    result["artifacts"] = [{"artifact_type": "report", "path": "report.txt"}]
    result["adapter_workspace_dir"] = str(workspace_dir)
    result["adapter_received_output_dir"] = "output_dir" in runtime
    return result


def settings() -> RunnerSettings:
    return RunnerSettings(
        port=0,
        runner_name="contract-test",
        runner_type="generator",
        runner_version="0.1.0",
        contract_version=1,
        idle_timeout_seconds=900,
        startup_timeout_seconds=60,
        adapter_target="unused:handler",
    )


class RunnerContractTests(unittest.TestCase):
    def test_request_validation(self) -> None:
        self.assertEqual(validate_job_request([]), "request body must be a JSON object")
        self.assertEqual(validate_job_request({}), "job must be an object")
        self.assertEqual(
            validate_job_request({"job": {"timeout_seconds": 1}}),
            "job.batch_id is required",
        )
        self.assertIn(
            "finite number",
            validate_job_request({"job": {"batch_id": "batch-1", "timeout_seconds": "nan"}}) or "",
        )
        self.assertEqual(
            validate_job_request(
                {"job": {"batch_id": "batch-1", "timeout_seconds": 1, "parameters": []}}
            ),
            "job.parameters must be an object",
        )
        self.assertIsNone(
            validate_job_request({"job": {"batch_id": "batch-1", "timeout_seconds": 1}})
        )

    def test_runner_rejects_a_different_batch(self) -> None:
        runner = Runner(settings=settings(), run_job_handler=completed_job, state="idle", batch_id="batch-1")
        accepted, error = runner.submit_job(
            {"job": {"job_id": "job-2", "batch_id": "batch-2", "timeout_seconds": 1}}
        )
        self.assertFalse(accepted)
        self.assertIn("does not match", error or "")

    def test_child_process_result_and_timeout(self) -> None:
        runner = Runner(settings=settings(), run_job_handler=completed_job, state="idle")
        request = {"job": {"job_id": "job-1", "batch_id": "batch-1", "timeout_seconds": 1}}
        result = runner._run_job_in_child_process(request, timeout_seconds=1, kill_after_seconds=1)
        self.assertEqual(result["status"], "completed")

        runner.run_job_handler = slow_job
        result = runner._run_job_in_child_process(request, timeout_seconds=0.01, kill_after_seconds=0.01)
        self.assertEqual(result["failure"]["code"], "RUNNER_JOB_TIMEOUT")

    def test_child_process_can_return_a_large_result(self) -> None:
        runner = Runner(settings=settings(), run_job_handler=large_result_job, state="idle")
        request = {"job": {"job_id": "job-1", "batch_id": "batch-1", "timeout_seconds": 2}}
        result = runner._run_job_in_child_process(request, timeout_seconds=2, kill_after_seconds=2)
        self.assertEqual(len(result["payload"]), 2 * 1024 * 1024)

    def test_child_process_stages_and_publishes_runtime_output(self) -> None:
        runner = Runner(settings=settings(), run_job_handler=output_job, state="idle")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "shared-output"
            request = {
                "job": {"job_id": "job-1", "batch_id": "batch-1", "timeout_seconds": 2},
                "runtime": {"output_dir": str(output_dir)},
            }
            result = runner._run_job_in_child_process(
                request,
                timeout_seconds=2,
                kill_after_seconds=2,
            )

            self.assertEqual(result["status"], "completed")
            self.assertNotEqual(Path(result["adapter_workspace_dir"]), output_dir)
            self.assertFalse(result["adapter_received_output_dir"])
            self.assertEqual((output_dir / "result.txt").read_text(), "published\n")
            self.assertEqual((output_dir / "report.txt").read_text(), "artifact\n")
            self.assertFalse((output_dir / "scratch.tmp").exists())

    def test_shutdown_terminates_active_job(self) -> None:
        runner = Runner(settings=settings(), run_job_handler=slow_job, state="idle")
        accepted, error = runner.submit_job(
            {"job": {"job_id": "job-1", "batch_id": "batch-1", "timeout_seconds": 10}}
        )
        self.assertTrue(accepted, error)

        deadline = time.time() + 2
        while runner.child_process is None and time.time() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(runner.child_process)

        runner.request_shutdown()
        deadline = time.time() + 2
        while runner.child_process is not None and time.time() < deadline:
            time.sleep(0.01)
        self.assertIsNone(runner.child_process)
        self.assertEqual(runner.state, "shutting_down")

    def test_http_rejects_non_object_json(self) -> None:
        runner = Runner(settings=settings(), run_job_handler=completed_job, state="idle")
        server = RunnerHTTPServer(("127.0.0.1", 0), RunnerHandler, runner)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("POST", "/run-job", body="[]", headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 400)
            self.assertFalse(payload["accepted"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
