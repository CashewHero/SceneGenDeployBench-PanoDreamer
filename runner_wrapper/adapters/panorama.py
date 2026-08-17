from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from runner_wrapper.adapters import perspective as shared
from runner_wrapper.job_logging import tee_job_output
from runner_wrapper.measurements import ResourceMonitor

logger = logging.getLogger("runner_wrapper.adapters.panorama")

RUNNER_NAME = "panodreamer-panorama"
TARGET_HEIGHT = 512
TARGET_WIDTH = round(
    2
    * math.pi
    * TARGET_HEIGHT
    / (2 * math.tan(math.radians(shared.MODEL_FOV_DEGREES / 2)))
)
TARGET_VERTICAL_FOV_DEGREES = math.degrees(
    2 * math.atan((TARGET_HEIGHT / 2) / (TARGET_WIDTH / (2 * math.pi)))
)
SUPPORTED_PROJECTIONS = {"cylindrical", "equirectangular"}
EQUIRECTANGULAR_ALIASES = {"equirect", "equirectangular", "latlong", "panorama", "spherical"}
CYLINDRICAL_ALIASES = {"cylinder", "cylindrical"}


def _variant_key(job_request: dict[str, Any]) -> str:
    job = job_request.get("job") if isinstance(job_request.get("job"), dict) else {}
    payload = {
        "primary_sample": job.get("primary_sample"),
        "parameters": job.get("parameters") or {},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()[:10]
    return f"panorama-{digest}"


def _normalize_parameters(raw_parameters: Any) -> dict[str, Any]:
    if raw_parameters is None:
        raw_parameters = {}
    if not isinstance(raw_parameters, dict):
        raise ValueError("job.parameters must be an object")

    return {
        "depth_method": shared._parameter_choice(
            raw_parameters, "depth_method", "dav2", {"dav2"}
        ),
        "depth_iterations": shared._parameter_int(
            raw_parameters, "depth_iterations", 15, minimum=1, maximum=100
        ),
        "depth_bins": shared._parameter_int(
            raw_parameters, "depth_bins", 10, minimum=2, maximum=100
        ),
        "metric_dataset": shared._parameter_choice(
            raw_parameters, "metric_dataset", "vkitti", {"hypersim", "vkitti"}
        ),
        "ldi_layers": shared._parameter_int(
            raw_parameters, "ldi_layers", 4, minimum=2, maximum=16
        ),
        "gs_iterations": shared._parameter_int(
            raw_parameters, "gs_iterations", 3000, minimum=1, maximum=100000
        ),
        "gs_views": shared._parameter_int(
            raw_parameters, "gs_views", 240, minimum=1, maximum=1440
        ),
        "gs_init_only": shared._parameter_bool(raw_parameters, "gs_init_only", False),
        "gs_scale_mult": shared._parameter_float(
            raw_parameters, "gs_scale_mult", 2.0, minimum=0.01, maximum=100.0
        ),
        "gs_init_opacity": shared._parameter_float(
            raw_parameters, "gs_init_opacity", 0.5, minimum=0.0001, maximum=0.9999
        ),
        "gs_freeze_positions": shared._parameter_bool(
            raw_parameters, "gs_freeze_positions", False
        ),
        "gs_depth_weight": shared._parameter_float(
            raw_parameters, "gs_depth_weight", 0.005, minimum=0.0, maximum=100.0
        ),
        "gs_novel_view_weight": shared._parameter_float(
            raw_parameters, "gs_novel_view_weight", 0.5, minimum=0.0, maximum=100.0
        ),
        "gs_novel_view_radius": shared._parameter_float(
            raw_parameters, "gs_novel_view_radius", 3.0, minimum=0.0, maximum=1000.0
        ),
        "gs_novel_view_start": shared._parameter_int(
            raw_parameters, "gs_novel_view_start", 500, minimum=1, maximum=100000
        ),
        "gs_novel_view_every": shared._parameter_int(
            raw_parameters, "gs_novel_view_every", 10, minimum=1, maximum=100000
        ),
        "gs_depth_max": shared._parameter_float(
            raw_parameters, "gs_depth_max", 200.0, minimum=0.1, maximum=100000.0
        ),
        "debug": shared._parameter_bool(raw_parameters, "debug", False),
    }


def _metadata_projection(metadata: Any) -> str:
    if metadata is None:
        return "equirectangular"
    if not isinstance(metadata, dict):
        raise ValueError("job.primary_sample_metadata must be an object when provided")
    projection = str(metadata.get("projection") or "").strip().lower()
    if not projection or projection in EQUIRECTANGULAR_ALIASES:
        return "equirectangular"
    if projection in CYLINDRICAL_ALIASES:
        return "cylindrical"
    if projection not in SUPPORTED_PROJECTIONS:
        allowed = ", ".join(sorted(SUPPORTED_PROJECTIONS))
        raise ValueError(
            f"panodreamer-panorama projection must be one of: {allowed}; "
            f"received {projection}"
        )
    return projection


def _inferred_vertical_fov(projection: str, size: tuple[int, int]) -> float:
    width, height = size
    if projection == "equirectangular":
        return min(180.0, 360.0 * height / width)
    return math.degrees(2 * math.atan(math.pi * height / width))


def _metadata_fov(
    metadata: dict[str, Any], projection: str, size: tuple[int, int]
) -> tuple[float, float]:
    raw_fov = metadata.get("fov")
    inferred_vertical = _inferred_vertical_fov(projection, size)
    if raw_fov in (None, ""):
        return 360.0, inferred_vertical

    if isinstance(raw_fov, dict):
        raw_horizontal = raw_fov.get("horizontal")
        raw_vertical = raw_fov.get("vertical")
    elif isinstance(raw_fov, (list, tuple)):
        raw_horizontal = raw_fov[0] if raw_fov else None
        raw_vertical = raw_fov[1] if len(raw_fov) > 1 else None
    else:
        raw_horizontal = raw_fov
        raw_vertical = None

    try:
        horizontal = 360.0 if raw_horizontal in (None, "") else float(raw_horizontal)
        vertical = inferred_vertical if raw_vertical in (None, "") else float(raw_vertical)
    except (TypeError, ValueError) as exc:
        raise ValueError("job.primary_sample_metadata.fov values must be numeric") from exc
    if not all(math.isfinite(value) and 0 < value <= 360 for value in (horizontal, vertical)):
        raise ValueError("job.primary_sample_metadata.fov values must be between 0 and 360")
    if vertical > 180.0 + 1e-3:
        raise ValueError("job.primary_sample_metadata vertical fov cannot exceed 180 degrees")
    return horizontal, vertical


def _validate_resolution_metadata(metadata: dict[str, Any], size: tuple[int, int]) -> None:
    raw_resolution = metadata.get("resolution")
    if raw_resolution in (None, ""):
        return
    if not isinstance(raw_resolution, (list, tuple)) or len(raw_resolution) != 2:
        logger.warning(
            shared.event_message(
                "panorama_resolution_metadata_ignored",
                reason="expected [width, height]",
                actual_size=list(size),
            )
        )
        return
    try:
        declared = (int(raw_resolution[0]), int(raw_resolution[1]))
    except (TypeError, ValueError):
        logger.warning(
            shared.event_message(
                "panorama_resolution_metadata_ignored",
                reason="resolution values were not integers",
                actual_size=list(size),
            )
        )
        return
    if declared != size:
        logger.warning(
            shared.event_message(
                "panorama_resolution_metadata_ignored",
                reason="declared resolution did not match the file",
                declared_size=list(declared),
                actual_size=list(size),
            )
        )


def _inspect_panorama(image_path: Path, metadata: Any) -> tuple[str, float, tuple[int, int]]:
    if not image_path.is_file():
        raise FileNotFoundError(f"input panorama not found: {image_path}")
    projection = _metadata_projection(metadata)
    normalized_metadata = metadata if isinstance(metadata, dict) else {}

    with Image.open(image_path) as source:
        source.verify()
        size = source.size
    _validate_resolution_metadata(normalized_metadata, size)

    horizontal_fov, vertical_fov = _metadata_fov(normalized_metadata, projection, size)
    if not math.isclose(horizontal_fov, 360.0, abs_tol=1.0):
        raise ValueError(
            "panodreamer-panorama requires a full 360 degree horizontal panorama; "
            f"received {horizontal_fov}"
        )

    if not normalized_metadata.get("projection") or normalized_metadata.get("fov") in (None, ""):
        logger.info(
            shared.event_message(
                "panorama_metadata_inferred",
                projection=projection,
                fov=[horizontal_fov, vertical_fov],
                image_size=list(size),
            )
        )

    if vertical_fov + 0.1 < TARGET_VERTICAL_FOV_DEGREES:
        raise ValueError(
            "input panorama vertical fov is too narrow for PanoDreamer: "
            f"requires {TARGET_VERTICAL_FOV_DEGREES:.3f} degrees, received {vertical_fov:.3f}"
        )
    return projection, vertical_fov, size


def _convert_to_cylindrical_cuda(
    image_path: Path,
    projection: str,
    vertical_fov_degrees: float,
    workspace_root: Path,
) -> Path:
    with Image.open(image_path) as source:
        source_size = source.size
    if (
        projection == "cylindrical"
        and source_size == (TARGET_WIDTH, TARGET_HEIGHT)
        and math.isclose(vertical_fov_degrees, TARGET_VERTICAL_FOV_DEGREES, abs_tol=0.1)
    ):
        return image_path

    import numpy as np
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("panorama projection conversion requires CUDA")

    with Image.open(image_path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        source_array = np.asarray(rgb).copy()

    source_tensor = (
        torch.from_numpy(source_array)
        .to(device="cuda", dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )
    source_height, source_width = source_array.shape[:2]
    target_focal = TARGET_WIDTH / (2 * math.pi)
    target_rows = torch.arange(TARGET_HEIGHT, device="cuda", dtype=torch.float32) + 0.5
    target_cols = torch.arange(TARGET_WIDTH, device="cuda", dtype=torch.float32) + 0.5
    elevation_down = torch.atan((target_rows - TARGET_HEIGHT / 2) / target_focal)
    grid_x = 2 * target_cols / TARGET_WIDTH - 1

    if projection == "equirectangular":
        vertical_fov_radians = math.radians(vertical_fov_degrees)
        grid_y = 2 * elevation_down / vertical_fov_radians
    else:
        source_focal = source_width / (2 * math.pi)
        grid_y = 2 * source_focal * torch.tan(elevation_down) / source_height

    grid = torch.stack(
        (
            grid_x.unsqueeze(0).expand(TARGET_HEIGHT, -1),
            grid_y.unsqueeze(1).expand(-1, TARGET_WIDTH),
        ),
        dim=-1,
    ).unsqueeze(0)
    converted = functional.grid_sample(
        source_tensor,
        grid,
        mode="bicubic",
        padding_mode="border",
        align_corners=False,
    ).clamp(0, 1)
    torch.cuda.synchronize()

    converted_array = (
        converted[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    )
    prepared_dir = workspace_root / "input"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prepared_dir / "cylindrical.png"
    Image.fromarray(converted_array, mode="RGB").save(prepared_path)
    logger.info(
        shared.event_message(
            "panorama_projection_converted",
            source_path=str(image_path),
            source_projection=projection,
            source_size=[source_width, source_height],
            source_vertical_fov_degrees=vertical_fov_degrees,
            target_path=str(prepared_path),
            target_projection="cylindrical",
            target_size=[TARGET_WIDTH, TARGET_HEIGHT],
            target_vertical_fov_degrees=TARGET_VERTICAL_FOV_DEGREES,
            device="cuda",
        )
    )
    return prepared_path


def _run_stage(
    stage: str,
    command: list[str],
    workspace_dir: Path,
    env: dict[str, str],
) -> None:
    logger.info(shared.event_message("model_stage_started", stage=stage, command=command))
    subprocess.run(command, check=True, cwd=workspace_dir, env=env)
    logger.info(shared.event_message("model_stage_finished", stage=stage))


def _failure_result(
    *, started_at: float, completed_at: float, stage: str, metrics: list[dict[str, Any]], log_name: str
) -> dict[str, Any]:
    return {
        "status": "failed",
        "started_at": shared.utc_time(started_at),
        "completed_at": shared.utc_time(completed_at),
        "metrics": metrics,
        "artifacts": [{"artifact_type": "job_log", "path": log_name}],
        "failure": {
            "code": "PANODREAMER_PANORAMA_RUN_FAILED",
            "message": f"PanoDreamer panorama pipeline failed during {stage}; see {log_name}",
            "retryable": False,
            "stage": "adapter",
        },
    }


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    workspace_root = Path(job_request["runtime"]["workspace_dir"])
    workspace_root.mkdir(parents=True, exist_ok=True)
    variant = _variant_key(job_request)
    log_path = workspace_root / f"runner-{variant}.log"
    with tee_job_output(log_path):
        return _run_job_logged(job_request, started_at, workspace_root, log_path, variant)


def _run_job_logged(
    job_request: dict[str, Any],
    started_at: float,
    workspace_root: Path,
    log_path: Path,
    variant: str,
) -> dict[str, Any]:
    monitor: ResourceMonitor | None = None
    resource_metrics: list[dict[str, Any]] = []
    stage = "validation"
    try:
        job = job_request["job"]
        inputs = shared._normalize_inputs(job_request.get("inputs"))
        primary_sample = str(job["primary_sample"])
        data_samples = inputs.get("data", {})
        if primary_sample not in data_samples:
            raise ValueError(f"primary sample not found in inputs.data: {primary_sample}")
        image_value = data_samples[primary_sample].get("image")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError("primary sample image must be a non-empty path")
        image_path = Path(image_value)
        parameters = _normalize_parameters(job.get("parameters"))
        projection, vertical_fov, _ = _inspect_panorama(
            image_path, job.get("primary_sample_metadata")
        )
        shared._ensure_model_sources()
        shared._require_cuda()

        monitor_data = {
            f"{role}.{sample_id}.{data_type}": value
            for role, samples in inputs.items()
            for sample_id, sample_values in samples.items()
            for data_type, value in sample_values.items()
        }
        monitor = ResourceMonitor(sample_data=monitor_data, output_dir=workspace_root)
        monitor.start()
        logger.info(
            shared.event_message(
                "adapter_run_started",
                job_id=job.get("job_id"),
                batch_id=job.get("batch_id"),
                workspace_dir=str(workspace_root),
                primary_sample=primary_sample,
                source_projection=projection,
            )
        )

        stage = "input_projection"
        panorama_path = _convert_to_cylindrical_cuda(
            image_path, projection, vertical_fov, workspace_root
        )

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TQDM_MININTERVAL", "10")
        cache_namespace = os.getenv("PANODREAMER_MODEL_CACHE_NAMESPACE", "panodreamer").strip()
        if not cache_namespace or Path(cache_namespace).name != cache_namespace:
            raise ValueError("PANODREAMER_MODEL_CACHE_NAMESPACE must be one directory name")
        model_cache_dir = Path(os.getenv("PATH_MODEL_CACHE", "/data/model_cache")) / cache_namespace
        download_dir = workspace_root / "model-downloads"
        dav2_checkpoint_dir = model_cache_dir / "depth-anything-v2"
        inpainting_checkpoint_dir = model_cache_dir / "inpainting_ckpts"
        env["PANODREAMER_DAV2_CHECKPOINT_DIR"] = str(dav2_checkpoint_dir)

        stage = "model_assets"
        shared._ensure_depth_weights(
            dav2_checkpoint_dir, download_dir, parameters["metric_dataset"]
        )
        shared._ensure_inpainting_weights(inpainting_checkpoint_dir, download_dir, env)
        (workspace_root / "inpainting_ckpts").symlink_to(
            inpainting_checkpoint_dir, target_is_directory=True
        )

        pipeline_dir = workspace_root / "pipeline"
        depth_dir = pipeline_dir / "depth"
        ldi_dir = pipeline_dir / "ldi"
        gs_dir = pipeline_dir / "gsplat"
        for path in (depth_dir, ldi_dir, gs_dir):
            path.mkdir(parents=True, exist_ok=True)

        stage = "depth_estimation"
        depth_command = [
            sys.executable,
            str(shared._repo_root() / "depth_estimation.py"),
            "--input_image", str(panorama_path),
            "--output_dir", str(depth_dir),
            "--mode", "panorama",
            "--iterations", str(parameters["depth_iterations"]),
            "--num_bins", str(parameters["depth_bins"]),
            "--fov", str(shared.MODEL_FOV_DEGREES),
            "--metric_dataset", parameters["metric_dataset"],
            "--method", parameters["depth_method"],
        ]
        if parameters["debug"]:
            depth_command.append("--debug")
        _run_stage(stage, depth_command, workspace_root, env)
        depth_path = depth_dir / "depth_pano.npy"
        if not shared._usable_file(depth_path):
            raise FileNotFoundError(f"depth stage did not produce {depth_path}")

        stage = "ldi_generation"
        ldi_command = [
            sys.executable,
            str(shared._repo_root() / "ldi_generation.py"),
            "--input_image", str(panorama_path),
            "--input_depth", str(depth_path),
            "--output_dir", str(ldi_dir),
            "--num_layers", str(parameters["ldi_layers"]),
            "--require_inpainter",
        ]
        if parameters["debug"]:
            ldi_command.append("--debug")
        _run_stage(stage, ldi_command, workspace_root, env)
        for filename in ("rgba_ldi.npy", "depth_ldi.npy", "mask_ldi.npy"):
            if not shared._usable_file(ldi_dir / filename):
                raise FileNotFoundError(f"LDI stage did not produce {ldi_dir / filename}")

        stage = "3dgs_optimization"
        source_ply = gs_dir / "scene.ply"
        gs_command = [
            sys.executable,
            str(shared._repo_root() / "train_gsplat.py"),
            "--ldi_dir", str(ldi_dir),
            "--output", str(source_ply),
            "--num_iterations", str(parameters["gs_iterations"]),
            "--num_views", str(parameters["gs_views"]),
            "--fov", str(shared.MODEL_FOV_DEGREES),
            "--video_interval", "0",
            "--scale_mult", str(parameters["gs_scale_mult"]),
            "--init_opacity", str(parameters["gs_init_opacity"]),
            "--depth_weight", str(parameters["gs_depth_weight"]),
            "--novel_view_weight", str(parameters["gs_novel_view_weight"]),
            "--novel_view_radius", str(parameters["gs_novel_view_radius"]),
            "--novel_view_start", str(parameters["gs_novel_view_start"]),
            "--novel_view_every", str(parameters["gs_novel_view_every"]),
            "--novel_view_model", "dav2",
            "--depth_max", str(parameters["gs_depth_max"]),
        ]
        if parameters["gs_init_only"]:
            gs_command.append("--init_only")
        if parameters["gs_freeze_positions"]:
            gs_command.append("--freeze_positions")
        if parameters["debug"]:
            gs_command.append("--debug")
        _run_stage(stage, gs_command, workspace_root, env)
        if not shared._usable_file(source_ply):
            raise FileNotFoundError(f"3DGS stage did not produce {source_ply}")

        output_name = f"3DGS-{variant}.ply"
        output_ply = workspace_root / output_name
        shutil.move(source_ply, output_ply)
        output_files = {primary_sample: {"3dgs": output_name}}

        resource_metrics = monitor.stop()
        monitor = None
        model_metrics = [
            {
                "namespace": "model",
                "name": "panorama_generation_skipped",
                "type": "boolean",
                "value": True,
                "source": "runner",
            },
            {
                "namespace": "model",
                "name": "depth_alignment_iterations",
                "type": "integer",
                "value": parameters["depth_iterations"],
                "unit": "iterations",
                "source": "model",
            },
        ]
        metrics = resource_metrics + model_metrics
        metrics_name = f"metrics-{variant}.json"
        shared._write_metrics_file(
            workspace_root / metrics_name,
            {
                "inputs": inputs,
                "output_files": output_files,
                "parameters": parameters,
                "source_projection": projection,
                "model_projection": "cylindrical",
                "model_metrics": model_metrics,
                **({"resource_metrics": resource_metrics} if resource_metrics else {}),
            },
        )

        completed_at = time.time()
        logger.info(
            shared.event_message(
                "adapter_run_completed",
                job_id=job.get("job_id"),
                output_ply=str(output_ply),
                wall_time_ms=round((completed_at - started_at) * 1000, 3),
            )
        )
        return {
            "status": "completed",
            "started_at": shared.utc_time(started_at),
            "completed_at": shared.utc_time(completed_at),
            "output_files": output_files,
            "metrics": metrics,
            "artifacts": [
                {"artifact_type": "job_log", "path": log_path.name},
                {"artifact_type": "metric_summary", "path": metrics_name},
            ],
            "failure": None,
        }
    except Exception as exc:
        if monitor is not None:
            resource_metrics = monitor.stop()
        completed_at = time.time()
        print(
            f"panodreamer-panorama job failed during {stage}: {exc}", file=sys.stderr, flush=True
        )
        traceback.print_exc()
        return _failure_result(
            started_at=started_at,
            completed_at=completed_at,
            stage=stage,
            metrics=resource_metrics,
            log_name=log_path.name,
        )
