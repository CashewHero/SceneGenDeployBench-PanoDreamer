from __future__ import annotations

import errno
import logging
import os
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger("runner_wrapper.files")


def publish_directory(
    source: str | Path,
    destination: str | Path,
    *,
    dirs_exist_ok: bool = False,
) -> Path:
    """Publish a directory atomically when new, or with atomic files when merging."""

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_dir():
        raise FileNotFoundError(f"source directory not found: {source_path}")
    if destination_path.exists() and not dirs_exist_ok:
        raise FileExistsError(destination_path)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        _copy_directory_contents(
            source_path,
            destination_path,
            atomic_files=True,
        )
        _sync_directory(destination_path)
        return destination_path

    staging_path = destination_path.with_name(
        f".{destination_path.name}.{uuid.uuid4().hex}.part"
    )
    try:
        staging_path.mkdir()
        _copy_directory_contents(
            source_path,
            staging_path,
            atomic_files=False,
        )
        _sync_directory(staging_path)
        try:
            os.rename(staging_path, destination_path)
        except OSError as exc:
            if not destination_path.exists():
                raise
            if not dirs_exist_ok:
                raise FileExistsError(destination_path) from exc
            _copy_directory_contents(
                staging_path,
                destination_path,
                atomic_files=True,
            )
            _sync_directory(destination_path)
        _sync_directory(destination_path.parent)
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)

    return destination_path


def _copy_directory_contents(
    source_path: Path,
    destination_path: Path,
    *,
    atomic_files: bool,
) -> None:
    directory_pairs: list[tuple[Path, Path]] = []
    preserve_metadata = True

    for root, directory_names, file_names in os.walk(source_path):
        directory_names.sort()
        file_names.sort()
        source_root = Path(root)
        destination_root = destination_path / source_root.relative_to(source_path)
        destination_root.mkdir(parents=True, exist_ok=True)
        directory_pairs.append((source_root, destination_root))

        for directory_name in directory_names:
            (destination_root / directory_name).mkdir(exist_ok=True)

        for file_name in file_names:
            source_entry = source_root / file_name
            destination_entry = destination_root / file_name
            if atomic_files:
                publish_file(source_entry, destination_entry)
            else:
                shutil.copyfile(source_entry, destination_entry)
                if preserve_metadata:
                    preserve_metadata = _preserve_metadata(
                        source_entry,
                        destination_entry,
                    )
                _sync_file(destination_entry)

    for source_dir, destination_dir in reversed(directory_pairs):
        if preserve_metadata:
            preserve_metadata = _preserve_metadata(source_dir, destination_dir)
        _sync_directory(destination_dir)


def publish_file(source: str | Path, destination: str | Path) -> Path:
    """Atomically copy a file, preserving metadata when the filesystem allows it."""

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"source file not found: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_name(
        f".{destination_path.name}.{uuid.uuid4().hex}.part"
    )
    try:
        shutil.copyfile(source_path, temp_path)
        _preserve_metadata(source_path, temp_path)
        _sync_file(temp_path)
        os.replace(temp_path, destination_path)
        _sync_directory(destination_path.parent)
    finally:
        temp_path.unlink(missing_ok=True)

    return destination_path


def _sync_file(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise
        # Publication still works on filesystems without fsync support.


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some remote and FUSE filesystems do not support directory fsync.
        pass
    finally:
        os.close(descriptor)


def _preserve_metadata(source: Path, destination: Path) -> bool:
    try:
        shutil.copystat(source, destination)
        return True
    except OSError as exc:
        logger.warning(
            "Could not preserve file metadata for %s: %s; keeping copied contents",
            destination,
            exc,
        )
        return False
