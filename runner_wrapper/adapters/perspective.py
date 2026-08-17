from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from runner_wrapper.files import publish_directory, publish_file
from runner_wrapper.job_logging import tee_job_output
from runner_wrapper.measurements import ResourceMonitor

logger = logging.getLogger("runner_wrapper.adapters.perspective")

RUNNER_NAME = "panodreamer-perspective"
MODEL_FOV_DEGREES = 44.701948991275390
DEFAULT_NEGATIVE_PROMPT = (
    "caption, subtitle, text, blur, lowres, bad anatomy, bad hands, cropped, "
    "worst quality, watermark"
)

DEPTH_CHECKPOINTS = {
    "relative": {
        "repo_id": "depth-anything/Depth-Anything-V2-Large",
        "filename": "depth_anything_v2_vitl.pth",
        "sha256": "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345",
    },
    "vkitti": {
        "repo_id": "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large",
        "filename": "depth_anything_v2_metric_vkitti_vitl.pth",
        "sha256": "239b1054a369e66da2576e9a118d6d7c12d90dc8ebe609579a9a09cd8e05fe38",
    },
    "hypersim": {
        "repo_id": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
        "filename": "depth_anything_v2_metric_hypersim_vitl.pth",
        "sha256": "6f82ff2bc543ac02ddff4aa31fa363676a8305dd3ccf04e80e2af115a044cb6d",
    },
}
INPAINTING_FILENAMES = ("depth-model.pth", "color-model.pth")


def event_message(event: str, **fields: object) -> str:
    return json.dumps({"event": event, **fields}, sort_keys=True)


def utc_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _variant_key(job_request: dict[str, Any]) -> str:
    job = job_request.get("job") if isinstance(job_request.get("job"), dict) else {}
    payload = {
        "primary_sample": job.get("primary_sample"),
        "parameters": job.get("parameters") or {},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()[:10]
    return f"perspective-{digest}"


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


def _parameter_int(
    values: dict[str, Any],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = values.get(name, default)
    if isinstance(raw, bool):
        raise ValueError(f"job.parameters.{name} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"job.parameters.{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"job.parameters.{name} must be between {minimum} and {maximum}")
    return value


def _parameter_float(
    values: dict[str, Any],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = values.get(name, default)
    if isinstance(raw, bool):
        raise ValueError(f"job.parameters.{name} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"job.parameters.{name} must be a number") from exc
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(f"job.parameters.{name} must be between {minimum} and {maximum}")
    return value


def _parameter_bool(values: dict[str, Any], name: str, default: bool) -> bool:
    raw = values.get(name, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"job.parameters.{name} must be a boolean")


def _parameter_choice(
    values: dict[str, Any],
    name: str,
    default: str,
    choices: set[str],
) -> str:
    value = str(values.get(name, default)).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"job.parameters.{name} must be one of: {allowed}")
    return value


def _normalize_parameters(raw_parameters: Any) -> dict[str, Any]:
    if raw_parameters is None:
        raw_parameters = {}
    if not isinstance(raw_parameters, dict):
        raise ValueError("job.parameters must be an object")

    prompt = str(raw_parameters.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("job.parameters.prompt must be a non-empty string")

    negative_prompt = str(raw_parameters.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT).strip()
    panorama_width = _parameter_int(
        raw_parameters, "panorama_width", 3912, minimum=512, maximum=8192
    )
    if panorama_width % 8:
        raise ValueError("job.parameters.panorama_width must be divisible by 8")

    parameters = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": _parameter_int(raw_parameters, "seed", 1234, minimum=0, maximum=2**31 - 1),
        "panorama_width": panorama_width,
        "panorama_steps": _parameter_int(
            raw_parameters, "panorama_steps", 50, minimum=1, maximum=200
        ),
        "panorama_iterations": _parameter_int(
            raw_parameters, "panorama_iterations", 15, minimum=1, maximum=100
        ),
        "guidance": _parameter_float(
            raw_parameters, "guidance", 7.5, minimum=0.0, maximum=50.0
        ),
        # This runner intentionally follows the repository's default DAV2 pipeline.
        "depth_method": _parameter_choice(raw_parameters, "depth_method", "dav2", {"dav2"}),
        "depth_iterations": _parameter_int(
            raw_parameters, "depth_iterations", 15, minimum=1, maximum=100
        ),
        "depth_bins": _parameter_int(raw_parameters, "depth_bins", 10, minimum=2, maximum=100),
        "metric_dataset": _parameter_choice(
            raw_parameters, "metric_dataset", "vkitti", {"hypersim", "vkitti"}
        ),
        "ldi_layers": _parameter_int(raw_parameters, "ldi_layers", 4, minimum=2, maximum=16),
        "gs_iterations": _parameter_int(
            raw_parameters, "gs_iterations", 3000, minimum=1, maximum=100000
        ),
        "gs_views": _parameter_int(raw_parameters, "gs_views", 240, minimum=1, maximum=1440),
        "gs_init_only": _parameter_bool(raw_parameters, "gs_init_only", False),
        "gs_scale_mult": _parameter_float(
            raw_parameters, "gs_scale_mult", 2.0, minimum=0.01, maximum=100.0
        ),
        "gs_init_opacity": _parameter_float(
            raw_parameters, "gs_init_opacity", 0.5, minimum=0.0001, maximum=0.9999
        ),
        "gs_freeze_positions": _parameter_bool(raw_parameters, "gs_freeze_positions", False),
        "gs_depth_weight": _parameter_float(
            raw_parameters, "gs_depth_weight", 0.005, minimum=0.0, maximum=100.0
        ),
        "gs_novel_view_weight": _parameter_float(
            raw_parameters, "gs_novel_view_weight", 0.5, minimum=0.0, maximum=100.0
        ),
        "gs_novel_view_radius": _parameter_float(
            raw_parameters, "gs_novel_view_radius", 3.0, minimum=0.0, maximum=1000.0
        ),
        "gs_novel_view_start": _parameter_int(
            raw_parameters, "gs_novel_view_start", 500, minimum=1, maximum=100000
        ),
        "gs_novel_view_every": _parameter_int(
            raw_parameters, "gs_novel_view_every", 10, minimum=1, maximum=100000
        ),
        "gs_depth_max": _parameter_float(
            raw_parameters, "gs_depth_max", 200.0, minimum=0.1, maximum=100000.0
        ),
        "debug": _parameter_bool(raw_parameters, "debug", False),
    }
    return parameters


def _input_horizontal_fov(metadata: Any) -> float | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("job.primary_sample_metadata must be an object")

    raw_fov = metadata.get("fov")
    if isinstance(raw_fov, (list, tuple)) and raw_fov:
        raw_fov = raw_fov[0]
    if raw_fov in (None, ""):
        return None
    try:
        fov = float(raw_fov)
    except (TypeError, ValueError) as exc:
        raise ValueError("job.primary_sample_metadata.fov must be numeric") from exc
    if not math.isfinite(fov) or fov <= 0.0 or fov >= 180.0:
        raise ValueError("job.primary_sample_metadata.fov must be between 0 and 180 degrees")
    return fov


def _validate_input_image(image_path: Path, metadata: Any) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")

    from PIL import Image

    with Image.open(image_path) as image:
        image.verify()
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        raise ValueError("job.primary_sample_metadata must be an object")
    projection = str(metadata.get("projection") or "").strip().lower()
    if projection and projection not in {"perspective", "pinhole"}:
        raise ValueError(
            "panodreamer-perspective requires a perspective/pinhole image; "
            f"received projection={projection}"
        )

    fov = _input_horizontal_fov(metadata)
    if fov is not None and fov < MODEL_FOV_DEGREES - 1.0:
        raise ValueError(
            "panodreamer-perspective requires at least a 44.702 degree horizontal FOV; "
            f"received {fov}"
        )


def _prepare_input_image(image_path: Path, metadata: Any, workspace_root: Path) -> Path:
    fov = _input_horizontal_fov(metadata)
    if fov is None or fov <= MODEL_FOV_DEGREES + 1.0:
        return image_path

    from PIL import Image, ImageOps

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        crop_ratio = math.tan(math.radians(MODEL_FOV_DEGREES / 2.0)) / math.tan(
            math.radians(fov / 2.0)
        )
        crop_size = round(image.width * crop_ratio)
        if crop_size < 1 or crop_size > image.width:
            raise ValueError(f"cannot crop input horizontal FOV from {fov} to {MODEL_FOV_DEGREES:.3f}")
        if image.height < crop_size:
            raise ValueError(
                "input image does not have enough vertical pixels for a square "
                f"{MODEL_FOV_DEGREES:.3f} degree crop"
            )

        left = (image.width - crop_size) // 2
        top = (image.height - crop_size) // 2
        crop_box = (left, top, left + crop_size, top + crop_size)
        prepared_dir = workspace_root / "input"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        prepared_path = prepared_dir / "fov-cropped.png"
        image.crop(crop_box).save(prepared_path)

    logger.info(
        event_message(
            "input_fov_cropped",
            input_path=str(image_path),
            prepared_path=str(prepared_path),
            source_fov_degrees=fov,
            model_fov_degrees=MODEL_FOV_DEGREES,
            crop_box=list(crop_box),
            crop_size=crop_size,
        )
    )
    return prepared_path


def _truthy_env(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"", "0", "false", "no", "off"}


def _usable_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


@contextmanager
def _exclusive_path_lock(target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_path.with_name(f".{target_path.name}.lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_depth_checkpoint(
    checkpoint: dict[str, str],
    checkpoint_dir: Path,
    download_dir: Path,
) -> Path:
    destination = checkpoint_dir / checkpoint["filename"]
    with _exclusive_path_lock(destination):
        if _usable_file(destination):
            return destination
        if not _truthy_env("PANODREAMER_AUTO_DOWNLOAD_WEIGHTS"):
            raise FileNotFoundError(
                f"missing Depth Anything checkpoint: {destination}; enable "
                "PANODREAMER_AUTO_DOWNLOAD_WEIGHTS or install the checkpoint"
            )

        from huggingface_hub import hf_hub_download

        logger.info(
            event_message(
                "depth_checkpoint_download_started",
                repo_id=checkpoint["repo_id"],
                filename=checkpoint["filename"],
            )
        )
        downloaded = Path(
            hf_hub_download(
                repo_id=checkpoint["repo_id"],
                filename=checkpoint["filename"],
                cache_dir=download_dir / "huggingface",
                token=os.getenv("HF_TOKEN") or None,
            )
        )
        actual_hash = _sha256(downloaded)
        if actual_hash != checkpoint["sha256"]:
            raise RuntimeError(
                f"checkpoint checksum mismatch for {checkpoint['filename']}: {actual_hash}"
            )
        publish_file(downloaded, destination)
        logger.info(
            event_message(
                "depth_checkpoint_download_finished",
                destination=str(destination),
                size_bytes=destination.stat().st_size,
            )
        )
    return destination


def _ensure_depth_weights(
    checkpoint_dir: Path,
    download_dir: Path,
    metric_dataset: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _ensure_depth_checkpoint(DEPTH_CHECKPOINTS["relative"], checkpoint_dir, download_dir)
    _ensure_depth_checkpoint(DEPTH_CHECKPOINTS[metric_dataset], checkpoint_dir, download_dir)


def _ensure_inpainting_weights(
    checkpoint_dir: Path,
    download_dir: Path,
    env: dict[str, str],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if all(_usable_file(checkpoint_dir / filename) for filename in INPAINTING_FILENAMES):
        return
    if not _truthy_env("PANODREAMER_AUTO_DOWNLOAD_WEIGHTS"):
        missing = [
            str(checkpoint_dir / filename)
            for filename in INPAINTING_FILENAMES
            if not _usable_file(checkpoint_dir / filename)
        ]
        raise FileNotFoundError(
            "missing PanoDreamer inpainting checkpoint(s): " + ", ".join(missing)
        )

    lock_target = checkpoint_dir / "inpainting-checkpoints"
    with _exclusive_path_lock(lock_target):
        if all(_usable_file(checkpoint_dir / filename) for filename in INPAINTING_FILENAMES):
            return
        local_dir = download_dir / f"inpainting-{uuid.uuid4().hex}"
        local_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(_repo_root() / "download_inpainting_ckpts.py"),
                "--output_dir",
                str(local_dir),
            ],
            check=True,
            cwd=_repo_root(),
            env=env,
        )
        for filename in INPAINTING_FILENAMES:
            matches = list(local_dir.rglob(filename))
            if len(matches) != 1 or not _usable_file(matches[0]):
                raise FileNotFoundError(
                    f"inpainting download did not produce exactly one usable {filename}"
                )
            publish_file(matches[0], checkpoint_dir / filename)


def _prepare_huggingface_cache(
    workspace_dir: Path,
    model_cache_dir: Path,
    env: dict[str, str],
) -> tuple[Path, Path, Path] | None:
    if env.get("HF_HOME"):
        return None
    shared_cache = model_cache_dir / "huggingface"
    ready_marker = shared_cache / ".stable-diffusion-ready"
    if ready_marker.is_file():
        env["HF_HOME"] = str(shared_cache)
        return None
    local_cache = workspace_dir / "model-downloads" / "huggingface"
    local_cache.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(local_cache)
    return local_cache, shared_cache, ready_marker


def _publish_huggingface_cache(cache_paths: tuple[Path, Path, Path] | None) -> None:
    if cache_paths is None:
        return
    local_cache, shared_cache, ready_marker = cache_paths
    if not local_cache.is_dir():
        raise FileNotFoundError(f"Hugging Face cache was not created: {local_cache}")
    with _exclusive_path_lock(ready_marker):
        publish_directory(local_cache, shared_cache, dirs_exist_ok=True)
        marker = local_cache.parent / ready_marker.name
        marker.write_text("ready\n", encoding="utf-8")
        publish_file(marker, ready_marker)


def _ensure_model_sources() -> None:
    required = (
        _repo_root() / "Depth-Anything-V2" / "depth_anything_v2" / "dpt.py",
        _repo_root() / "3d-moments" / "core" / "inpainter.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing model submodule file(s): "
            + ", ".join(missing)
            + "; initialize repository submodules"
        )


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("panodreamer-perspective requires CUDA")


def _run_stage(
    stage: str,
    command: list[str],
    workspace_dir: Path,
    env: dict[str, str],
) -> None:
    logger.info(event_message("model_stage_started", stage=stage, command=command))
    subprocess.run(command, check=True, cwd=workspace_dir, env=env)
    logger.info(event_message("model_stage_finished", stage=stage))


def _write_metrics_file(metrics_path: Path, summary: dict[str, Any]) -> None:
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def _failure_result(
    *,
    started_at: float,
    completed_at: float,
    stage: str,
    metrics: list[dict[str, Any]],
    log_name: str,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "started_at": utc_time(started_at),
        "completed_at": utc_time(completed_at),
        "metrics": metrics,
        "artifacts": [{"artifact_type": "job_log", "path": log_name}],
        "failure": {
            "code": "PANODREAMER_PERSPECTIVE_RUN_FAILED",
            "message": f"PanoDreamer perspective pipeline failed during {stage}; see {log_name}",
            "retryable": False,
            "stage": "adapter",
        },
    }


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    runtime = job_request["runtime"]
    workspace_root = Path(runtime["workspace_dir"])
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
        inputs = _normalize_inputs(job_request.get("inputs"))
        data_samples = inputs.get("data", {})
        primary_sample = str(job["primary_sample"])
        if primary_sample not in data_samples:
            raise ValueError(f"primary sample not found in inputs.data: {primary_sample}")
        sample_data = data_samples[primary_sample]
        image_value = sample_data.get("image")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError("primary sample image must be a non-empty path")
        image_path = Path(image_value)
        parameters = _normalize_parameters(job.get("parameters"))
        _validate_input_image(image_path, job.get("primary_sample_metadata"))
        _ensure_model_sources()
        _require_cuda()

        monitor_data = {
            f"{role}.{sample_id}.{data_type}": value
            for role, samples in inputs.items()
            for sample_id, sample_values in samples.items()
            for data_type, value in sample_values.items()
        }
        monitor = ResourceMonitor(sample_data=monitor_data, output_dir=workspace_root)
        monitor.start()
        logger.info(
            event_message(
                "adapter_run_started",
                job_id=job.get("job_id"),
                batch_id=job.get("batch_id"),
                workspace_dir=str(workspace_root),
                primary_sample=primary_sample,
            )
        )

        stage = "input_preprocessing"
        model_input_path = _prepare_input_image(
            image_path,
            job.get("primary_sample_metadata"),
            workspace_root,
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
        hf_cache_paths = _prepare_huggingface_cache(workspace_root, model_cache_dir, env)

        stage = "model_assets"
        _ensure_depth_weights(
            dav2_checkpoint_dir,
            download_dir,
            parameters["metric_dataset"],
        )
        _ensure_inpainting_weights(inpainting_checkpoint_dir, download_dir, env)
        inpainting_link = workspace_root / "inpainting_ckpts"
        inpainting_link.symlink_to(inpainting_checkpoint_dir, target_is_directory=True)

        pipeline_dir = workspace_root / "pipeline"
        panorama_root = pipeline_dir / "panorama"
        depth_dir = pipeline_dir / "depth"
        ldi_dir = pipeline_dir / "ldi"
        gs_dir = pipeline_dir / "gsplat"
        for path in (panorama_root, depth_dir, ldi_dir, gs_dir):
            path.mkdir(parents=True, exist_ok=True)
        prompt_path = pipeline_dir / "prompt.txt"
        prompt_path.write_text(parameters["prompt"] + "\n", encoding="utf-8")

        stage = "panorama_generation"
        panorama_command = [
            sys.executable,
            str(_repo_root() / "multicondiffusion_panorama.py"),
            "--prompt_file",
            str(prompt_path),
            "--input_image",
            str(model_input_path),
            "--negative",
            parameters["negative_prompt"],
            "--H",
            "512",
            "--W",
            str(parameters["panorama_width"]),
            "--seed",
            str(parameters["seed"]),
            "--steps",
            str(parameters["panorama_steps"]),
            "--iterations",
            str(parameters["panorama_iterations"]),
            "--guidance",
            str(parameters["guidance"]),
            "--output_dir",
            str(panorama_root),
        ]
        if parameters["debug"]:
            panorama_command.append("--debug")
        _run_stage(stage, panorama_command, workspace_root, env)
        _publish_huggingface_cache(hf_cache_paths)

        panorama_path = (
            panorama_root
            / f"prompt_seed{parameters['seed']}"
            / "final_output_prompt.png"
        )
        if not _usable_file(panorama_path):
            raise FileNotFoundError(f"panorama stage did not produce {panorama_path}")

        stage = "depth_estimation"
        depth_command = [
            sys.executable,
            str(_repo_root() / "depth_estimation.py"),
            "--input_image",
            str(panorama_path),
            "--output_dir",
            str(depth_dir),
            "--mode",
            "panorama",
            "--iterations",
            str(parameters["depth_iterations"]),
            "--num_bins",
            str(parameters["depth_bins"]),
            "--fov",
            str(MODEL_FOV_DEGREES),
            "--metric_dataset",
            parameters["metric_dataset"],
            "--method",
            parameters["depth_method"],
        ]
        if parameters["debug"]:
            depth_command.append("--debug")
        _run_stage(stage, depth_command, workspace_root, env)
        depth_path = depth_dir / "depth_pano.npy"
        if not _usable_file(depth_path):
            raise FileNotFoundError(f"depth stage did not produce {depth_path}")

        stage = "ldi_generation"
        ldi_command = [
            sys.executable,
            str(_repo_root() / "ldi_generation.py"),
            "--input_image",
            str(panorama_path),
            "--input_depth",
            str(depth_path),
            "--output_dir",
            str(ldi_dir),
            "--num_layers",
            str(parameters["ldi_layers"]),
            "--require_inpainter",
        ]
        if parameters["debug"]:
            ldi_command.append("--debug")
        _run_stage(stage, ldi_command, workspace_root, env)
        for filename in ("rgba_ldi.npy", "depth_ldi.npy", "mask_ldi.npy"):
            if not _usable_file(ldi_dir / filename):
                raise FileNotFoundError(f"LDI stage did not produce {ldi_dir / filename}")

        stage = "3dgs_optimization"
        source_ply = gs_dir / "scene.ply"
        gs_command = [
            sys.executable,
            str(_repo_root() / "train_gsplat.py"),
            "--ldi_dir",
            str(ldi_dir),
            "--output",
            str(source_ply),
            "--num_iterations",
            str(parameters["gs_iterations"]),
            "--num_views",
            str(parameters["gs_views"]),
            "--fov",
            str(MODEL_FOV_DEGREES),
            "--video_interval",
            "0",
            "--scale_mult",
            str(parameters["gs_scale_mult"]),
            "--init_opacity",
            str(parameters["gs_init_opacity"]),
            "--depth_weight",
            str(parameters["gs_depth_weight"]),
            "--novel_view_weight",
            str(parameters["gs_novel_view_weight"]),
            "--novel_view_radius",
            str(parameters["gs_novel_view_radius"]),
            "--novel_view_start",
            str(parameters["gs_novel_view_start"]),
            "--novel_view_every",
            str(parameters["gs_novel_view_every"]),
            "--novel_view_model",
            "dav2",
            "--depth_max",
            str(parameters["gs_depth_max"]),
        ]
        if parameters["gs_init_only"]:
            gs_command.append("--init_only")
        if parameters["gs_freeze_positions"]:
            gs_command.append("--freeze_positions")
        if parameters["debug"]:
            gs_command.append("--debug")
        _run_stage(stage, gs_command, workspace_root, env)
        if not _usable_file(source_ply):
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
                "name": "inference_steps",
                "type": "integer",
                "value": parameters["panorama_steps"] * parameters["panorama_iterations"],
                "unit": "steps",
                "source": "model",
            }
        ]
        metrics = resource_metrics + model_metrics
        metrics_name = f"metrics-{variant}.json"
        metrics_path = workspace_root / metrics_name
        report: dict[str, Any] = {
            "inputs": inputs,
            "output_files": output_files,
            "parameters": parameters,
            "model_metrics": model_metrics,
        }
        if resource_metrics:
            report["resource_metrics"] = resource_metrics
        _write_metrics_file(metrics_path, report)

        completed_at = time.time()
        logger.info(
            event_message(
                "adapter_run_completed",
                job_id=job.get("job_id"),
                output_ply=str(output_ply),
                wall_time_ms=round((completed_at - started_at) * 1000, 3),
            )
        )
        return {
            "status": "completed",
            "started_at": utc_time(started_at),
            "completed_at": utc_time(completed_at),
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
            f"panodreamer-perspective job failed during {stage}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return _failure_result(
            started_at=started_at,
            completed_at=completed_at,
            stage=stage,
            metrics=resource_metrics,
            log_name=log_path.name,
        )
