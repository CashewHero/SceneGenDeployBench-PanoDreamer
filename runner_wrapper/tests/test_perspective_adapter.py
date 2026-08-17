from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from runner_wrapper.adapters import perspective


class ParameterTests(unittest.TestCase):
    def test_prompt_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt"):
            perspective._normalize_parameters({})

    def test_official_depth_method_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "depth_method"):
            perspective._normalize_parameters({"prompt": "a room", "depth_method": "moge"})

    def test_panorama_width_must_be_divisible_by_eight(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            perspective._normalize_parameters({"prompt": "a room", "panorama_width": 1001})


class InputValidationTests(unittest.TestCase):
    def test_rejects_equirectangular_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.png"
            Image.new("RGB", (512, 512)).save(image_path)
            with self.assertRaisesRegex(ValueError, "perspective/pinhole"):
                perspective._validate_input_image(image_path, {"projection": "equirectangular"})

    def test_accepts_non_square_perspective_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.png"
            Image.new("RGB", (640, 480)).save(image_path)
            perspective._validate_input_image(image_path, {"projection": "pinhole"})

    def test_center_crops_wide_fov_input_for_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "image.png"
            Image.new("RGB", (640, 640)).save(image_path)

            perspective._validate_input_image(
                image_path,
                {"projection": "pinhole", "fov": [90.0, 90.0]},
            )
            prepared_path = perspective._prepare_input_image(
                image_path,
                {"projection": "pinhole", "fov": [90.0, 90.0]},
                root / "workspace",
            )

            self.assertNotEqual(prepared_path, image_path)
            with Image.open(prepared_path) as prepared:
                self.assertEqual(prepared.size, (263, 263))

    def test_rejects_input_narrower_than_model_fov(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.png"
            Image.new("RGB", (512, 512)).save(image_path)
            with self.assertRaisesRegex(ValueError, "at least a 44.702 degree"):
                perspective._validate_input_image(
                    image_path,
                    {"projection": "pinhole", "fov": 30.0},
                )


class PerspectiveAdapterPipelineTests(unittest.TestCase):
    def test_runs_official_four_stage_pipeline_and_publishes_3dgs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "input.png"
            workspace = root / "workspace"
            Image.new("RGB", (512, 512), color=(20, 40, 60)).save(image_path)
            stages: list[str] = []

            def fake_stage(
                stage: str,
                command: list[str],
                workspace_dir: Path,
                env: dict[str, str],
            ) -> None:
                del workspace_dir, env
                stages.append(stage)

                def value(flag: str) -> Path:
                    return Path(command[command.index(flag) + 1])

                if stage == "panorama_generation":
                    output_dir = value("--output_dir") / "prompt_seed1234"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (512, 1024)).save(output_dir / "final_output_prompt.png")
                elif stage == "depth_estimation":
                    output_dir = value("--output_dir")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "depth_pano.npy").write_bytes(b"depth")
                elif stage == "ldi_generation":
                    output_dir = value("--output_dir")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    for filename in ("rgba_ldi.npy", "depth_ldi.npy", "mask_ldi.npy"):
                        (output_dir / filename).write_bytes(b"ldi")
                elif stage == "3dgs_optimization":
                    output = value("--output")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"ply")

            request = {
                "contract_version": 1,
                "job": {
                    "job_id": "job-1",
                    "batch_id": "batch-1",
                    "job_type": "generation",
                    "primary_sample": "sample-1",
                    "primary_sample_metadata": {
                        "projection": "pinhole",
                        "fov": perspective.MODEL_FOV_DEGREES,
                    },
                    "attempt": 1,
                    "timeout_seconds": 3600,
                    "parameters": {
                        "prompt": "a realistic room",
                        "panorama_width": 1024,
                        "panorama_steps": 1,
                        "panorama_iterations": 1,
                        "depth_iterations": 1,
                        "ldi_layers": 2,
                        "gs_iterations": 1,
                        "gs_views": 1,
                        "gs_init_only": True,
                    },
                },
                "inputs": {"data": {"sample-1": {"image": str(image_path)}}},
                "runtime": {"workspace_dir": str(workspace)},
            }

            with (
                patch.object(perspective, "_ensure_model_sources"),
                patch.object(perspective, "_require_cuda"),
                patch.object(perspective, "_ensure_depth_weights"),
                patch.object(perspective, "_ensure_inpainting_weights"),
                patch.object(perspective, "_prepare_huggingface_cache", return_value=None),
                patch.object(perspective, "_run_stage", side_effect=fake_stage),
            ):
                result = perspective.run_job(request)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                stages,
                [
                    "panorama_generation",
                    "depth_estimation",
                    "ldi_generation",
                    "3dgs_optimization",
                ],
            )
            output_name = result["output_files"]["sample-1"]["3dgs"]
            self.assertTrue((workspace / output_name).is_file())
            self.assertEqual(result["metrics"][-1]["name"], "inference_steps")


if __name__ == "__main__":
    unittest.main()
