from __future__ import annotations

"""Default adapter implementation.

Replace this file or point RUNNER_ADAPTER at a different callable inside the
model repository.
"""

import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from runner_wrapper.job_logging import tee_job_output
from runner_wrapper.measurements import ResourceMonitor

logger = logging.getLogger("runner_wrapper.adapter")


def event_message(event: str, **fields: object) -> str:
    return json.dumps({"event": event, **fields}, sort_keys=True)


def _safe_role(role: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in role)


def _safe_name(value: Any, fallback: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value or "").strip()).strip("-._")
    return name or fallback


def _parameter_variant(parameters: Any) -> str:
    if not isinstance(parameters, dict) or not parameters:
        return "default"
    encoded = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:10]
    readable = "variant"
    for _, value in sorted(parameters.items()):
        if isinstance(value, bool):
            value = str(value).lower()
        if isinstance(value, (str, int, float)):
            readable = _safe_name(value, "variant")[:24]
            break
    return f"{readable}-{digest}"


def _normalize_inputs(raw_inputs: Any) -> dict[str, dict[str, dict[str, Any]]]:
    if raw_inputs is None:
        return {}
    if not isinstance(raw_inputs, dict):
        raise ValueError("inputs must be an object")

    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_role, raw_samples in raw_inputs.items():
        role = str(raw_role).strip()
        if not role or not isinstance(raw_samples, dict):
            raise ValueError("each input role must contain a sample mapping")
        samples: dict[str, dict[str, Any]] = {}
        for raw_sample_id, raw_sample_data in raw_samples.items():
            sample_id = str(raw_sample_id).strip()
            if not sample_id or not isinstance(raw_sample_data, dict):
                raise ValueError(f"inputs.{role} must map sample ids to data mappings")
            sample_data: dict[str, Any] = {}
            for raw_data_type, value in raw_sample_data.items():
                data_type = str(raw_data_type).strip()
                if not data_type:
                    raise ValueError(f"inputs.{role}.{sample_id} contains an empty data type")
                sample_data[data_type] = value.strip() if isinstance(value, str) else value
            if sample_data:
                samples[sample_id] = sample_data
        if samples:
            normalized[role] = samples
    return normalized


def _copy_inputs(
    samples: dict[str, dict[str, Any]],
    workspace_root: Path,
    variant: str,
) -> tuple[dict[str, dict[str, str]], int]:
    output_files: dict[str, dict[str, str]] = {}
    copied = 0
    for sample_index, (sample_id, sample_data) in enumerate(samples.items()):
        sample_outputs: dict[str, str] = {}
        for data_index, (data_type, raw_path) in enumerate(sample_data.items()):
            if not isinstance(raw_path, str):
                continue
            src_path = Path(raw_path)
            if not src_path.exists() or not src_path.is_file():
                raise FileNotFoundError(f"input file not found: {src_path}")
            dst_name = (
                f"input-{sample_index:02d}-{data_index:02d}-"
                f"{_safe_role(sample_id)}-{_safe_role(data_type)}-"
                f"{variant}{src_path.suffix}"
            )
            dst_path = workspace_root / dst_name
            shutil.copyfile(src_path, dst_path)
            sample_outputs[data_type] = str(dst_path.relative_to(workspace_root))
            copied += 1
        if sample_outputs:
            output_files[sample_id] = sample_outputs
    return output_files, copied


def _sleep_range_seconds() -> int:
    min_seconds = int(os.getenv("TEST_RUNNER_MIN_SECONDS", "360"))
    max_seconds = int(os.getenv("TEST_RUNNER_MAX_SECONDS", "720"))
    if min_seconds < 0 or max_seconds < min_seconds:
        raise ValueError("invalid TEST_RUNNER_MIN_SECONDS / TEST_RUNNER_MAX_SECONDS")
    return random.randint(min_seconds, max_seconds)


def _write_metrics_file(metrics_path: Path, summary: dict[str, Any]) -> None:
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def _is_evaluator_mode() -> bool:
    runner_type = os.getenv("RUNNER_TYPE", "generator").strip().lower()
    mode = os.getenv("TEST_RUNNER_MODE", "").strip().lower()
    return runner_type == "evaluator" or mode == "evaluator"


def _random_evaluation_metrics() -> list[dict[str, Any]]:
    return [
        {
            "namespace": "quality",
            "name": "test_quality_score",
            "type": "float",
            "value": round(random.uniform(0.0, 1.0), 6),
            "unit": "score",
            "source": "evaluator",
        },
        {
            "namespace": "quality",
            "name": "test_geometry_error",
            "type": "float",
            "value": round(random.uniform(0.0, 0.25), 6),
            "unit": "normalized_error",
            "source": "evaluator",
        },
    ]


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    runtime = job_request["runtime"]
    workspace_root = Path(runtime["workspace_dir"])
    workspace_root.mkdir(parents=True, exist_ok=True)
    parameters = job_request.get("job", {}).get("parameters") or {}
    variant = _parameter_variant(parameters)
    log_path = workspace_root / f"runner-{variant}.log"
    with tee_job_output(log_path):
        return _run_job_logged(
            job_request,
            started_at,
            workspace_root,
            log_path,
            variant,
        )


def _run_job_logged(
    job_request: dict[str, Any],
    started_at: float,
    workspace_root: Path,
    log_path: Path,
    variant: str,
) -> dict[str, Any]:
    monitor: ResourceMonitor | None = None

    try:
        job = job_request["job"]
        runtime = job_request["runtime"]
        inputs = _normalize_inputs(job_request.get("inputs"))
        data_samples = inputs.get("data", {})
        monitor_data = {
            f"{role}.{sample_id}.{data_type}": value
            for role, samples in inputs.items()
            for sample_id, sample_data in samples.items()
            for data_type, value in sample_data.items()
        }
        monitor = ResourceMonitor(sample_data=monitor_data, output_dir=workspace_root)
        monitor.start()
        logger.info(
            event_message(
                "adapter_run_started",
                job_id=job["job_id"],
                batch_id=job.get("batch_id"),
                workspace_dir=runtime["workspace_dir"],
                input_roles=sorted(inputs),
            )
        )

        metrics_path = workspace_root / f"metrics-{variant}.json"
        print(f"test runner job {job['job_id']} started", flush=True)

        sleep_seconds = _sleep_range_seconds()
        logger.info(event_message("adapter_sleeping", job_id=job["job_id"], sleep_seconds=sleep_seconds))
        print(f"test runner sleeping for {sleep_seconds} seconds", flush=True)

        time.sleep(sleep_seconds)
        evaluator_mode = _is_evaluator_mode()
        output_files: dict[str, dict[str, str]] = {}
        copied_input_count = 0
        if not evaluator_mode:
            print("test runner copying data inputs", flush=True)
            output_files, copied_input_count = _copy_inputs(
                data_samples,
                workspace_root,
                variant,
            )
        logger.info(
            event_message(
                "adapter_inputs_copied",
                job_id=job["job_id"],
                copied_input_count=copied_input_count,
            )
        )

        evaluation_metrics = _random_evaluation_metrics() if evaluator_mode else []

        resource_metrics = monitor.stop()
        monitor = None
        metrics = resource_metrics + evaluation_metrics
        completed_at = time.time()
        wall_time_ms = round((completed_at - started_at) * 1000, 3)
        print(f"test runner copied {copied_input_count} output files", flush=True)
        print(f"test runner completed in {wall_time_ms} ms", flush=True)
        report: dict[str, Any] = {"inputs": inputs}
        if output_files:
            report["output_files"] = output_files
        if job.get("parameters"):
            report["parameters"] = dict(job["parameters"])
        if evaluation_metrics:
            report["metrics"] = evaluation_metrics
        if resource_metrics:
            report["resource_metrics"] = resource_metrics
        _write_metrics_file(metrics_path, report)

        logger.info(
            event_message(
                "adapter_run_completed",
                job_id=job["job_id"],
                wall_time_ms=wall_time_ms,
                copied_input_count=copied_input_count,
            )
        )

        result = {
            "status": "completed",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(completed_at)),
            "metrics": metrics,
            "artifacts": [
                {
                    "artifact_type": "job_log",
                    "path": log_path.name,
                },
                {
                    "artifact_type": "metric_summary",
                    "path": metrics_path.name,
                },
            ],
            "failure": None,
        }
        if output_files:
            result["output_files"] = output_files
        return result
    except Exception as exc:
        resource_metrics = monitor.stop() if monitor is not None else []
        completed_at = time.time()
        print(f"test runner job failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return {
            "status": "failed",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(completed_at)),
            "metrics": resource_metrics,
            "artifacts": [
                {
                    "artifact_type": "job_log",
                    "path": log_path.name,
                }
            ],
            "failure": {
                "code": "TEST_RUNNER_FAILED",
                "message": f"Test runner failed; see {log_path.name}",
                "retryable": False,
                "stage": "adapter",
            },
        }
