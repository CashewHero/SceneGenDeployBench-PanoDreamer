from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    value = _read_text(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _file_tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _cpu_usage_usec() -> int | None:
    cpu_stat = _read_text(Path("/sys/fs/cgroup/cpu.stat"))
    if cpu_stat:
        for line in cpu_stat.splitlines():
            key, _, value = line.partition(" ")
            if key == "usage_usec":
                return int(value)

    usage_ns = _read_int(Path("/sys/fs/cgroup/cpuacct/cpuacct.usage"))
    if usage_ns is not None:
        return usage_ns // 1000
    return None


def _memory_current_bytes() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        value = _read_int(path)
        if value is not None:
            return value
    return None


def _io_stats() -> dict[str, int] | None:
    io_stat = _read_text(Path("/sys/fs/cgroup/io.stat"))
    if not io_stat:
        return None
    totals = {"rbytes": 0, "wbytes": 0, "rios": 0, "wios": 0}
    for line in io_stat.splitlines():
        for token in line.split()[1:]:
            key, _, value = token.partition("=")
            if key in totals:
                totals[key] += int(value)
    return totals


def _descendant_pids(root_pid: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = stat_path.read_text(encoding="utf-8")
            pid = int(stat_path.parent.name)
            ppid = int(raw.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(pid)

    found = {root_pid}
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def _nvidia_smi(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _gpu_device_memory_total_bytes() -> int | None:
    output = _nvidia_smi(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    if not output:
        return None
    total_mib = 0
    for line in output.splitlines():
        try:
            total_mib += int(line.strip())
        except ValueError:
            continue
    return total_mib * 1024 * 1024 if total_mib else None


def _gpu_process_memory_bytes(root_pid: int) -> int | None:
    output = _nvidia_smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    if not output:
        return None

    pids = _descendant_pids(root_pid)
    total_mib = 0
    matched = False
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            used_mib = int(parts[1])
        except ValueError:
            continue
        if pid in pids:
            matched = True
            total_mib += used_mib
    return total_mib * 1024 * 1024 if matched else None


def _metric(namespace: str, name: str, value: int | float, unit: str, metric_type: str = "integer") -> dict[str, Any]:
    return {
        "namespace": namespace,
        "name": name,
        "type": metric_type,
        "value": value,
        "unit": unit,
        "source": "runner",
    }


class ResourceMonitor:
    def __init__(
        self,
        *,
        sample_data: dict[str, Any],
        output_dir: Path,
        sample_interval_seconds: float = 0.5,
    ) -> None:
        self.sample_data = sample_data
        self.output_dir = output_dir
        self.sample_interval_seconds = sample_interval_seconds
        self.root_pid = os.getpid()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        self._wall_start = 0.0
        self._cpu_start: int | None = None
        self._io_start: dict[str, int] | None = None
        self._peak_memory_bytes: int | None = None
        self._gpu_peak_memory_bytes: int | None = None
        self._gpu_total_memory_bytes: int | None = None
        self._input_total_bytes = 0

    def start(self) -> None:
        self._started = True
        self._wall_start = time.time()
        self._cpu_start = _cpu_usage_usec()
        self._io_start = _io_stats()
        self._input_total_bytes = sum(
            _file_tree_size(Path(path))
            for path in self.sample_data.values()
            if isinstance(path, str)
        )
        self._gpu_total_memory_bytes = _gpu_device_memory_total_bytes()
        self._sample_once()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        if not self._started or self._stopped:
            return []
        self._stopped = True
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval_seconds + 1.0)
        self._sample_once()

        wall_time_ms = round((time.time() - self._wall_start) * 1000, 3)
        metrics: list[dict[str, Any]] = [_metric("performance", "wall_time_ms", wall_time_ms, "ms", "float")]

        cpu_end = _cpu_usage_usec()
        if self._cpu_start is not None and cpu_end is not None:
            metrics.append(_metric("resources", "cpu_time_ms", round((cpu_end - self._cpu_start) / 1000, 3), "ms", "float"))

        if self._peak_memory_bytes is not None:
            metrics.append(_metric("resources", "peak_memory_bytes", self._peak_memory_bytes, "bytes"))

        io_end = _io_stats()
        if self._io_start is not None and io_end is not None:
            metrics.extend(
                [
                    _metric("resources", "disk_read_bytes", max(io_end["rbytes"] - self._io_start["rbytes"], 0), "bytes"),
                    _metric("resources", "disk_write_bytes", max(io_end["wbytes"] - self._io_start["wbytes"], 0), "bytes"),
                    _metric("resources", "disk_read_ops", max(io_end["rios"] - self._io_start["rios"], 0), "ops"),
                    _metric("resources", "disk_write_ops", max(io_end["wios"] - self._io_start["wios"], 0), "ops"),
                ]
            )

        metrics.append(_metric("resources", "input_total_bytes", self._input_total_bytes, "bytes"))
        metrics.append(_metric("resources", "output_total_bytes", _file_tree_size(self.output_dir), "bytes"))

        if self._gpu_peak_memory_bytes is not None:
            metrics.append(_metric("resources", "gpu_peak_memory_bytes", self._gpu_peak_memory_bytes, "bytes"))
        if self._gpu_total_memory_bytes is not None:
            metrics.append(_metric("gpu", "device_memory_total_bytes", self._gpu_total_memory_bytes, "bytes"))

        return metrics

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        memory_current = _memory_current_bytes()
        if memory_current is not None:
            self._peak_memory_bytes = max(self._peak_memory_bytes or 0, memory_current)

        gpu_current = _gpu_process_memory_bytes(self.root_pid)
        if gpu_current is not None:
            self._gpu_peak_memory_bytes = max(self._gpu_peak_memory_bytes or 0, gpu_current)
