# Runner Adaptation Instructions

These instructions are for coding agents adapting this wrapper inside a model repository.

## Scope And Boundaries

Verify the repository naming requirement documented in the README before editing. Stop and notify the user if it is not satisfied.

Wrap the existing model rather than redesigning it. Prefer changes in `runner_wrapper/`, dependency and Docker wiring, catalog configuration, and small launch scripts. Modify original model code only when integration cannot be achieved cleanly from the wrapper, and explain why in the handoff.

Follow the runner-role boundary defined in `README.md`.

Reuse `job_logging.py` and `measurements.py`. Keep `server.py` stable unless the shared HTTP contract itself changes.

Do not assume that an orchestrator source tree exists in the target repository. Do not write to PostgreSQL. Do not put private credentials, datasets, caches, or local model weights in the image or repository.

## Inspect Before Editing

Identify:

1. the real inference entry point;
2. required Python, system, CUDA, and model dependencies;
3. how weights are found or downloaded;
4. accepted model inputs and their coordinate/projection assumptions;
5. generated files or evaluator scores;
6. the smallest realistic smoke input.

Choose stable semantic data types from the benchmark domain, such as `image`, `depth`, `camera_pose`, `camera_trajectory`, `3dgs`, `mesh`, `scene`, or `point_cloud`. Do not use model-local variable names as contract types.

## Adaptation Sequence

1. Choose one runner role and its semantic input and output types.
2. Replace the bundled test logic in `adapter.py` with the smallest model-specific integration.
3. Update `Dockerfile` and the repository dependencies for the model. Copy `examples/dockerignore.example` to the repository root as `.dockerignore` if an equivalent file is not already present.
4. Update the matching request example into a realistic smoke request. Copy the matching catalog example to `runner_wrapper/config/runners/<runner>.yaml`, then make both agree with the adapter.
5. Add a short note to the model repository's main README naming the runner role, semantic inputs/outputs, and where to find the wrapper instructions.
6. Add the image workflow using the README command when the repository will publish through GitHub Actions.
7. Run the unit, build, and smoke checks before handoff.

The bundled adapter and examples are contract fixtures, not a complete model implementation. The test adapter supports both fixture roles only so this shared wrapper can be tested. Replace placeholder names, image tags, paths, data types, generated files, and evaluator metrics. Preserve the linked runner contract and the adaptation rules below.

## Implement The Contract

Implement the canonical [Runner API](docs/api.md), including lifecycle, request roles, runtime paths, results, and failure behavior. Do not replace the shared lifecycle in `server.py`.

Read [Camera Pose](docs/camera_pose.md), [Camera Trajectory](docs/camera_trajectory.md), and [Depth Inputs](docs/depth.md) when those types are used.

Device access and runner-specific paths or flags belong in container and catalog environment configuration.

## Metrics And Logging

Use `ResourceMonitor` around the model job and report available standard measurements:

```text
resources.cpu_time_ms
resources.peak_memory_bytes
resources.disk_read_bytes
resources.disk_write_bytes
resources.disk_read_ops
resources.disk_write_ops
resources.input_total_bytes
resources.output_total_bytes
resources.gpu_peak_memory_bytes
gpu.device_memory_total_bytes
performance.wall_time_ms
```

Optional model measurements may include `model.estimated_ops`, `model.inference_steps`, `gpu.energy_joules`, or `gpu.compute_time_ms`. Omit values that cannot be measured; never report guessed zeroes.

Evaluator metric entries use stable `namespace`, `name`, `type`, `value`, optional `unit`, and `source` fields. Metric `type` is `float`, `integer`, `boolean`, or `string`; `source` is normally `runner`, `model`, or `evaluator`. Keep evaluator quality scores separate from resource metrics.

Use `tee_job_output` so model stdout and stderr reach both Docker logs and `runner-<variant>.log`. Preserve exceptions and useful progress while avoiding high-frequency progress-bar noise.

## Shared Filesystem Publication

Use `runtime.workspace_dir` for all job-local runtime files. Return files that need to be kept as relative paths in `output_files` or `artifacts`; `server.py` publishes them to `runtime.output_dir`. Never write to `runtime.output_dir` directly. Prepare model-cache entries locally, then publish them with `publish_file` or `publish_directory` from `runner_wrapper.files`.

## Catalog Alignment

Create one YAML under `runner_wrapper/config/runners/` from the matching example. This is the runner's distributable catalog; a deployment copies it into its active runner-config directory. Do not make the model repository depend on an orchestrator checkout. Ensure these fields match the adapter:

- required identity: `runner`, `version`, `display_name`, and `kind`
- version selection: `latest` and `contract_version`
- `inputs.data`
- `inputs.candidate`
- `inputs.references`
- `job_parameters`
- `launcher.driver` and `launcher.compat_version`
- `launcher.image` for the Docker driver
- `launcher.endpoint.port`
- `launcher.env` and `launcher.env_passthrough`
- optional Docker settings such as `launcher.gpus` and `launcher.user`
- `scheduling.max_batch_size`, `max_attempts`, `job_timeout_minutes`, and `startup_timeout_minutes`

Catalog input config example:

```yaml
inputs:
  data:
    required_sample:
      required_datatype: [image]
      optional_datatype: [camera_pose]
  candidate:
    required_sample:
      required_datatype: [3dgs]
  references:
    optional_sample:
      required_datatype: [image, camera_pose]
```

Set `kind` explicitly to one supported role. Mark exactly one version of a runner name as `latest` when multiple versions are present. Keep `contract_version` aligned with the wrapper server.

Names must line up exactly:

```text
catalog inputs.data       -> request inputs.data
catalog inputs.candidate  -> request inputs.candidate
catalog inputs.references -> request inputs.references
producer output_files     -> downstream evaluator inputs.candidate
```

The semantic data type reported by a producer must be the type required by its consumer. The catalog describes inputs; a producer reports the outputs it actually created through each job result's `output_files` mapping.

## Docker Adaptation

Keep heavyweight or mutable weights in the shared model cache unless licensing or reproducibility requires image-bundled public assets. Make automatic downloads concurrency-safe and deterministic. Document required tokens without committing them.

If the runner will have a large image build, change the github image workflow cache from `mode=max` to `mode=min`.

Use the catalog `launcher.env` for model mode, checkpoint selection, thresholds, backend flags, and paths. Use `env_passthrough` only for values supplied by deployment, such as credentials.

The HTTP process must become ready without loading per-job inputs. Load expensive reusable model state at startup only when that model benefits from it and failures remain clear.

## Verification

Run the build and smoke commands documented in the README, then verify:

- the server reaches `idle`;
- a realistic small request is accepted;
- the job reaches `finished` or returns a useful failure;
- runner log and metrics JSON files exist;
- reusable files appear only in `output_files` with correct semantic types;
- evaluator metrics are scalar and stable;
- paths work inside the container without host-only assumptions;
- the catalog image, port, role, types, and environment match the built image.

## Handoff

Report:

- the selected role and semantic input/output types;
- required model assets and environment variables;
- build and smoke results;
- any changes outside `runner_wrapper/` and why they were necessary;
- any remaining limitation that affects real benchmark runs.
