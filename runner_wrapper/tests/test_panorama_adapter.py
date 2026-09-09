from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from runner_wrapper.adapters import panorama


class PanoramaMetadataTests(unittest.TestCase):
    def test_missing_metadata_defaults_to_full_equirectangular(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "panorama.png"
            Image.new("RGB", (1024, 512)).save(image_path)
            projection, vertical_fov, size = panorama._inspect_panorama(image_path, None)
            self.assertEqual(projection, "equirectangular")
            self.assertEqual(vertical_fov, 180)
            self.assertEqual(size, (1024, 512))

    def test_missing_fov_uses_equirectangular_aspect_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "panorama.png"
            Image.new("RGB", (1024, 256)).save(image_path)
            projection, vertical_fov, _ = panorama._inspect_panorama(image_path, {})
            self.assertEqual(projection, "equirectangular")
            self.assertEqual(vertical_fov, 90)

    def test_missing_vertical_fov_is_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "panorama.png"
            Image.new("RGB", (1024, 512)).save(image_path)
            projection, vertical_fov, _ = panorama._inspect_panorama(
                image_path, {"projection": "equirectangular", "fov": [360]}
            )
            self.assertEqual(projection, "equirectangular")
            self.assertEqual(vertical_fov, 180)

    def test_stale_resolution_metadata_uses_actual_image_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "panorama.png"
            Image.new("RGB", (1024, 512)).save(image_path)
            _, _, size = panorama._inspect_panorama(
                image_path, {"projection": "equirectangular", "resolution": [2048, 1024]}
            )
            self.assertEqual(size, (1024, 512))

    def test_accepts_full_equirectangular_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "panorama.png"
            Image.new("RGB", (1024, 512)).save(image_path)
            projection, vertical_fov, size = panorama._inspect_panorama(
                image_path,
                {
                    "projection": "equirectangular",
                    "resolution": [1024, 512],
                    "fov": [360, 180],
                },
            )
            self.assertEqual(projection, "equirectangular")
            self.assertEqual(vertical_fov, 180)
            self.assertEqual(size, (1024, 512))

    def test_infers_native_cylindrical_vertical_fov(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "panorama.png"
            Image.new("RGB", (panorama.TARGET_WIDTH, panorama.TARGET_HEIGHT)).save(image_path)
            projection, vertical_fov, _ = panorama._inspect_panorama(
                image_path, {"projection": "cylindrical"}
            )
            self.assertEqual(projection, "cylindrical")
            self.assertAlmostEqual(vertical_fov, panorama.TARGET_VERTICAL_FOV_DEGREES)

    def test_rejects_partial_panorama(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "panorama.png"
            Image.new("RGB", (1024, 512)).save(image_path)
            with self.assertRaisesRegex(ValueError, "full 360"):
                panorama._inspect_panorama(
                    image_path,
                    {"projection": "equirectangular", "fov": [180, 90]},
                )

    def test_native_cylindrical_input_is_not_reencoded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "panorama.png"
            Image.new("RGB", (panorama.TARGET_WIDTH, panorama.TARGET_HEIGHT)).save(image_path)
            result = panorama._convert_to_cylindrical_cuda(
                image_path,
                "cylindrical",
                panorama.TARGET_VERTICAL_FOV_DEGREES,
                root / "workspace",
            )
            self.assertEqual(result, image_path)


class PanoramaAdapterPipelineTests(unittest.TestCase):
    def test_skips_generation_and_runs_three_downstream_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "input.png"
            workspace = root / "workspace"
            Image.new("RGB", (1024, 512), color=(20, 40, 60)).save(image_path)
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

                if stage == "depth_estimation":
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
                        "projection": "equirectangular",
                        "resolution": [1024, 512],
                        "fov": [360, 180],
                    },
                    "attempt": 1,
                    "timeout_seconds": 3600,
                    "parameters": {
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
            prepared_path = root / "prepared.png"
            Image.new("RGB", (panorama.TARGET_WIDTH, panorama.TARGET_HEIGHT)).save(prepared_path)

            with (
                patch.object(panorama.shared, "_ensure_model_sources"),
                patch.object(panorama.shared, "_require_cuda"),
                patch.object(panorama.shared, "_ensure_depth_weights"),
                patch.object(panorama.shared, "_ensure_inpainting_weights"),
                patch.object(
                    panorama, "_convert_to_cylindrical_cuda", return_value=prepared_path
                ),
                patch.object(panorama, "_run_stage", side_effect=fake_stage),
            ):
                result = panorama.run_job(request)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                stages, ["depth_estimation", "ldi_generation", "3dgs_optimization"]
            )
            output_name = result["output_files"]["sample-1"]["3dgs"]
            self.assertTrue((workspace / output_name).is_file())
            self.assertEqual(
                result["output_metadata"],
                {"scene_scale": 1.4, "scene_coordinate_system": "LDB"},
            )
            self.assertNotIn("prompt", result)
            self.assertEqual(result["metrics"][-2]["name"], "panorama_generation_skipped")


if __name__ == "__main__":
    unittest.main()
