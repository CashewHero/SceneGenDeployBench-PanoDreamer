from __future__ import annotations

import errno
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runner_wrapper.files import (
    publish_directory,
    publish_file,
)


class PublishFileTests(unittest.TestCase):
    def test_preserves_metadata_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            destination = root / "output" / "destination.bin"
            source.write_bytes(b"scene data")
            source.chmod(0o640)
            timestamp_ns = 1_700_000_000_123_456_789
            os.utime(source, ns=(timestamp_ns, timestamp_ns))

            publish_file(source, destination)

            self.assertEqual(destination.read_bytes(), b"scene data")
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o640)
            self.assertEqual(destination.stat().st_mtime_ns, timestamp_ns)

    def test_publishes_contents_when_metadata_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            destination = root / "output" / "destination.bin"
            source.write_bytes(b"scene data")

            error = PermissionError(errno.EPERM, "Operation not permitted")
            with (
                mock.patch("runner_wrapper.files.shutil.copystat", side_effect=error),
                self.assertLogs("runner_wrapper.files", level="WARNING"),
            ):
                publish_file(source, destination)

            self.assertEqual(destination.read_bytes(), b"scene data")
            self.assertFalse(any(destination.parent.glob("*.part")))

    def test_replace_failure_preserves_destination_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            destination = root / "output" / "destination.bin"
            source.write_bytes(b"new data")
            destination.parent.mkdir()
            destination.write_bytes(b"old data")

            with (
                mock.patch(
                    "runner_wrapper.files.os.replace",
                    side_effect=OSError("replace failed"),
                ),
                self.assertRaises(OSError),
            ):
                publish_file(source, destination)

            self.assertEqual(destination.read_bytes(), b"old data")
            self.assertFalse(any(destination.parent.glob("*.part")))

    def test_sync_failure_preserves_destination_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            destination = root / "output" / "destination.bin"
            source.write_bytes(b"new data")
            destination.parent.mkdir()
            destination.write_bytes(b"old data")

            error = OSError(errno.EIO, "sync failed")
            with (
                mock.patch("runner_wrapper.files.os.fsync", side_effect=error),
                self.assertRaises(OSError),
            ):
                publish_file(source, destination)

            self.assertEqual(destination.read_bytes(), b"old data")
            self.assertFalse(any(destination.parent.glob("*.part")))


class PublishDirectoryTests(unittest.TestCase):
    def test_preserves_contents_and_metadata_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            destination = root / "output"
            source.mkdir()
            sample = source / "sample.bin"
            sample.write_bytes(b"dataset")
            sample.chmod(0o640)
            timestamp_ns = 1_700_000_000_123_456_789
            os.utime(sample, ns=(timestamp_ns, timestamp_ns))

            publish_directory(source, destination)

            copied = destination / "sample.bin"
            self.assertEqual(copied.read_bytes(), b"dataset")
            self.assertEqual(stat.S_IMODE(copied.stat().st_mode), 0o640)
            self.assertEqual(copied.stat().st_mtime_ns, timestamp_ns)

    def test_keeps_contents_when_metadata_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            destination = root / "output"
            (source / "nested").mkdir(parents=True)
            (source / "nested" / "sample.bin").write_bytes(b"dataset")

            error = PermissionError(errno.EPERM, "Operation not permitted")
            with (
                mock.patch("runner_wrapper.files.shutil.copystat", side_effect=error),
                self.assertLogs("runner_wrapper.files", level="WARNING") as logs,
            ):
                publish_directory(source, destination)

            self.assertEqual((destination / "nested" / "sample.bin").read_bytes(), b"dataset")
            self.assertEqual(len(logs.output), 1)

    def test_copy_failure_does_not_expose_partial_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            destination = root / "output"
            source.mkdir()
            (source / "sample.bin").write_bytes(b"dataset")

            with (
                mock.patch(
                    "runner_wrapper.files.shutil.copyfile",
                    side_effect=OSError("copy failed"),
                ),
                self.assertRaises(OSError),
            ):
                publish_directory(source, destination)

            self.assertFalse(destination.exists())
            self.assertFalse(any(root.glob(".output.*.part")))

    def test_merge_uses_atomic_file_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            destination = root / "output"
            source.mkdir()
            destination.mkdir()
            (source / "new.bin").write_bytes(b"new")
            (destination / "existing.bin").write_bytes(b"existing")

            with mock.patch(
                "runner_wrapper.files.publish_file",
                wraps=publish_file,
            ) as publish:
                publish_directory(source, destination, dirs_exist_ok=True)

            publish.assert_called_once_with(
                source / "new.bin",
                destination / "new.bin",
            )
            self.assertEqual((destination / "new.bin").read_bytes(), b"new")
            self.assertEqual(
                (destination / "existing.bin").read_bytes(),
                b"existing",
            )
