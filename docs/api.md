# Runner API

HTTP JSON contract between the orchestrator and a runner.

`adapter.py` implements:

```python
def run_job(job_request: dict) -> dict:
    ...
```

## Model

- one runner instance serves one adapter
- one runner instance binds to one `batch_id`
- one job runs at a time
- `POST /run-job` returns after accept/reject; execution continues in the runner
- terminal result is returned through `GET /status`

## Endpoints

- `GET /status`
- `POST /run-job`
- `POST /shutdown`

## States

- `starting`
- `idle`
- `running`
- `finished`
- `failed`
- `shutting_down`

## `GET /status`

Fields:

- `online`
- `runner_name`
- `runner_type`
- `runner_version`
- `contract_version`
- `batch_id`
- `state`
- `current_job_id`
- `updated_at`
- `result`

`result` is present only for `finished` or `failed`.

## `POST /run-job`

Accepted from:

- `idle`
- `finished`
- `failed`

Rejected when:

- runner is `starting`, `running`, or `shutting_down`
- `job.batch_id` is missing
- `job.batch_id` does not match the runner's bound batch

Request shape:

```json
{
  "contract_version": 1,
  "job": {
    "job_id": "job-123",
    "batch_id": "batch-1",
    "job_type": "generation",
    "primary_sample": "sample-1",
    "primary_sample_metadata": {"projection": "equirectangular"},
    "source_job_id": null,
    "attempt": 1,
    "timeout_seconds": 3600,
    "parameters": {}
  },
  "inputs": {
    "data": {
      "sample-1": {
        "image": "/data/datasets/testset1/image/frame_0001.png"
      }
    }
  },
  "runtime": {
    "output_dir": "/data/output/test-runner@0.1.0/testset1/scene_a/frame_0001"
  }
}
```

The request has exactly four top-level fields: `contract_version`, `job`, `inputs`, and `runtime`. It does not include the catalog contract or a root `config` object.

The orchestrator validates catalog requirements before dispatch. Adapter validation covers operational assumptions such as readable files, supported formats, and model constraints.

### Job

- `job_id`: durable orchestrator job identity
- `batch_id`: batch served by this runner instance
- `job_type`: runner role selected by the catalog
- `primary_sample`: input sample that owns the job
- `primary_sample_metadata`: inherited dataset metadata; omitted when empty
- `primary_output_metadata`: metadata reported by the job that produced the primary generated input; omitted when empty
- `source_job_id`: upstream job identity when applicable
- `attempt`: current retry attempt
- `timeout_seconds`: required adapter timeout
- `parameters`: catalog defaults merged with per-job overrides

### Inputs

Every role has the same shape:

```text
inputs -> role -> sample_id -> data_type -> data
```

- `data`: selected dataset samples or reusable outputs used as normal input
- `candidate`: reusable upstream outputs selected for evaluation
- `references`: additional dataset or output samples selected for comparison

`candidate` is the result being evaluated; it does not describe files produced by the current runner. File and directory values are absolute paths. Structured types such as `camera_pose` remain JSON values. Empty roles are omitted.

The CLI `--dataset`, `--candidate`, and `--reference` selectors populate these roles. Each selector accepts dataset or output targets. With only a candidate, the orchestrator reuses the candidate job's original `inputs.data`.

### Runtime And Environment

`runtime.output_dir` is the only per-job runtime path in the request and is the final destination for durable job files. Use `PATH_MODEL_CACHE` for reusable downloaded assets.

Runner containers receive shared `PATH_DATASETS`, `PATH_MODEL_CACHE`, `PATH_OUTPUT`, and read-only `PATH_PIPELINES` variables. The wrapper also injects `RUNNER_PORT`, `RUNNER_NAME`, `RUNNER_TYPE`, `RUNNER_VERSION`, `RUNNER_CONTRACT_VERSION`, and `RUNNER_STARTUP_TIMEOUT_SECONDS`. The image configures `RUNNER_ADAPTER` and `RUNNER_IDLE_TIMEOUT_SECONDS`; the latter stops an inactive server after status polling ends. Use these values through `server.py`; do not hardcode a second identity or port in the adapter.

### Execution

An accepted response includes `accepted: true` and `state: "running"`.

`job.timeout_seconds` is required. The adapter runs in a child process, and the server terminates it if it is still running after `job.timeout_seconds + 60` seconds. The startup watchdog uses `RUNNER_STARTUP_TIMEOUT_SECONDS`, normally the catalog startup timeout plus one minute.

## `POST /shutdown`

- always returns `accepted: true`
- sets state to `shutting_down`
- shuts down the HTTP server after responding

## Result

```json
{
  "status": "completed",
  "started_at": "2026-04-18T10:00:00Z",
  "completed_at": "2026-04-18T10:07:31Z",
  "output_files": {
    "sample-1": {
      "3dgs": "3DGS-high-a1b2c3d4.ply"
    }
  },
  "output_metadata": {"scene_scale": 1.0},
  "metrics": [],
  "artifacts": [
    {"artifact_type": "job_log", "path": "runner-high-a1b2c3d4.log"},
    {"artifact_type": "metric_summary", "path": "metrics-high-a1b2c3d4.json"}
  ],
  "failure": null
}
```

Failure shape:

```json
{
  "status": "failed",
  "started_at": "2026-04-18T10:00:00Z",
  "completed_at": "2026-04-18T10:00:05Z",
  "metrics": [],
  "artifacts": [],
  "failure": {
    "code": "MODEL_ERROR",
    "message": "reason",
    "retryable": false,
    "stage": "adapter"
  }
}
```

## Result Rules

- `output_files` uses `sample_id -> data_type -> relative path` and is omitted when the runner produces no reusable files.
- `output_metadata` describes the generated outputs as a group and is omitted when empty. `scene_scale` is a nonzero finite number: displacement from the primary viewpoint in dataset space is multiplied by it before rendering in the generated scene; a negative value reverses the displacement direction.
- The orchestrator resolves and stores the sample/data-type mapping in `output_files`; output targets can reuse it in any input role.
- Construct `output_files` explicitly; do not infer it by scanning the output directory.
- Keep semantic output keys such as `image`, `3dgs`, or `mesh`; uniqueness belongs in the filename.
- `artifacts` contains logs and reports, not reusable outputs.
- Output and artifact paths are relative to `runtime.output_dir`.
- `metrics` contains evaluator scores and standard measurements.
- Durable job history belongs to the orchestrator, not the runner.

Construct `output_files` once, write the same mapping near the top of the human-readable metrics JSON, and return it in the result. Keep the output folder flat where practical. Parameter variants may share an output directory, so use a short readable value in each filename, such as `scene-high-a1b2c3d4.ply`, `runner-high-a1b2c3d4.log`, and `metrics-high-a1b2c3d4.json`.

Recommended metrics JSON order:

1. normalized `inputs`;
2. non-empty `output_files`;
3. non-empty `parameters`;
4. evaluator `metrics` and standard `resource_metrics`, when present.

For handled errors, return the failure result shape. Uncaught adapter exceptions are converted by `server.py` into runner failures.
