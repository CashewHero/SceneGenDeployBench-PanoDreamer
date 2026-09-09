# Runner Wrapper

This repository has two generator adapters that produce a `3dgs` PLY. `panodreamer-perspective` runs the full PanoDreamer pipeline from a perspective/pinhole `image` plus `job.parameters.prompt`; input metadata may declare a horizontal `fov`, with wider views center-cropped to 44.702 degrees and narrower views rejected. `panodreamer-panorama` starts at the depth stage from a full-360 panorama `image`, accepts `cylindrical` or `equirectangular`, and has no prompt or diffusion parameters.

The panorama adapter uses PanoDreamer's native full-360 cylindrical canvas of 3912x512 pixels with approximately 44.702 degrees of vertical coverage. A matching cylindrical input is passed through. Other supported panorama geometry is resampled with PyTorch on CUDA; equirectangular latitude is mapped to cylindrical height rather than resized. Missing projection metadata defaults to equirectangular, the most common panorama format. Missing FOV defaults to 360 degrees horizontally and infers vertical coverage from image aspect ratio; for example, 2:1 becomes 360x180 degrees. A cylindrical image with a declared projection but no FOV is treated as a full-360 square-pixel cylindrical projection. Missing or stale resolution metadata uses the file's actual dimensions. Explicit partial panoramas, insufficient vertical coverage, pinhole, fisheye, and unspecified cubemap layouts are rejected.

`runner_wrapper/` turns a model repository into a SceneGenDeployBench runner image. It provides the HTTP server, job logging, resource measurements, Docker wiring, examples, and local test helper. Model-specific entry points live under `adapters/`; each runner catalog entry selects exactly one.

The directory `runner_wrapper/` is self-contained so it can be copied or pulled as a subtree without the main repository.

One runner catalog entry has one role:

- A generator turns dataset inputs into reusable generated files.
- An evaluator consumes dataset data and/or files from a generator and reports metrics.

## Add To A Model Repository

From the model repository root:

```bash
git remote add deploybench https://github.com/CashewHero/SceneGenDeployBench.git
git fetch deploybench subtree/runner_wrapper
git subtree add --prefix=runner_wrapper deploybench subtree/runner_wrapper --squash
```

Pull later updates with:

```bash
git fetch deploybench subtree/runner_wrapper
git subtree pull --prefix=runner_wrapper deploybench subtree/runner_wrapper --squash
```

The main files are:

```text
runner_wrapper/
  adapters/        model-specific job implementations
    perspective.py perspective image pipeline
    panorama.py    panorama-input pipeline
  files.py         compatible artifact publication
  server.py        shared HTTP runner server
  Dockerfile       runner image build
  localtest.sh     local build and smoke helper
  AGENTS.md        detailed adaptation contract
  examples/        request, catalog, Docker, and workflow templates
```

The distributable `runner_wrapper/config/runners/panodreamer.yaml` catalog contains both runners. Copy it into the active DeployBench runner-config directory.

## Build And Test

Initialize the pinned model source dependencies and build from the model repository root:

```bash
git submodule update --init --recursive
docker build -f runner_wrapper/Dockerfile -t scenegendeploybench-panodreamer:local .
```

Or use the helper:

```bash
runner_wrapper/localtest.sh build
runner_wrapper/localtest.sh smoke
```

The smoke request uses the included campus image and deliberately reduced model settings. The first run downloads several large public checkpoints into `data/model_cache`; set `HF_TOKEN` when the deployment requires authenticated Hugging Face access.

## Data Flow

The orchestrator supplies the selected dataset data to a runner. An evaluator can also receive generated files and additional dataset viewpoints. Each runner reports reusable outputs or metrics back to the orchestrator.

The wire contract is defined in [Runner API](docs/api.md). Use the wrapper filesystem helpers to publish job files.

## Publish An Image

Create the image workflow from the included template:

```bash
mkdir -p .github/workflows
cp runner_wrapper/examples/github-workflows/build-runner-image.yaml \
  .github/workflows/runner-image.yaml
```

The target repository should be named `SceneGenDeployBench-<model>`. The workflow derives the GHCR image name from the repository name.
