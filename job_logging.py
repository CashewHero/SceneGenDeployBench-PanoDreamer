from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Iterator


PROGRESS_LOG_INTERVAL_SECONDS = 300.0
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROGRESS_LINE_RE = re.compile(
    r"^(?P<label>.*?)\s*(?P<percent>\d{1,3})%\|.*\|\s*\d+/(?P<total>\d+)(?:\s|$)"
)


class _TeeStream:
    def __init__(
        self,
        console: Any,
        log_file: Any,
        lock: threading.Lock,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.console = console
        self.log_file = log_file
        self.lock = lock
        self.clock = clock
        self.progress_buffer: str | None = None
        self.last_progress_text: str | None = None
        self.last_progress_log_time: float | None = None
        self.line_progress: dict[tuple[str, int], tuple[float, str, str]] = {}

    def write(self, data: str) -> int:
        with self.lock:
            self.console.write(data)
            self.console.flush()
            log_data = self._throttle_progress_lines(self._log_data(data))
            if log_data:
                self.log_file.write(log_data)
                self.log_file.flush()
        return len(data)

    def _log_data(self, data: str) -> str:
        if (
            self.progress_buffer is not None
            and "\r" not in data
            and "\n" in data
            and data.strip("\n")
        ):
            self.progress_buffer = None
            return data

        output: list[str] = []
        start = 0
        for index, character in enumerate(data):
            if character not in {"\r", "\n"}:
                continue
            text = data[start:index]
            if self.progress_buffer is None:
                output.append(text)
            else:
                self.progress_buffer += text

            if character == "\r":
                self.progress_buffer = ""
            elif self.progress_buffer is not None:
                progress = self._progress_line(force=True)
                if progress:
                    output.append(progress)
                self.progress_buffer = None
            else:
                output.append("\n")
            start = index + 1

        text = data[start:]
        if self.progress_buffer is None:
            output.append(text)
        else:
            self.progress_buffer += text
            progress = self._progress_line(force=False)
            if progress:
                output.append(progress)
        return "".join(output)

    def _throttle_progress_lines(self, data: str) -> str:
        output: list[str] = []
        for line in data.splitlines(keepends=True):
            if not line.endswith(("\n", "\r")):
                output.append(line)
                continue
            text = _ANSI_ESCAPE_RE.sub("", line).rstrip("\r\n")
            match = _PROGRESS_LINE_RE.match(text)
            if match is None or "Warning:" in text:
                output.append(line)
                continue

            key = (match.group("label").strip(), int(match.group("total")))
            now = self.clock()
            state = self.line_progress.get(key)
            if state is None or now - state[0] >= PROGRESS_LOG_INTERVAL_SECONDS:
                output.append(text + "\n")
                self.line_progress[key] = (now, text, text)
            else:
                self.line_progress[key] = (state[0], state[1], text)
        return "".join(output)

    def _progress_line(self, *, force: bool) -> str:
        assert self.progress_buffer is not None
        text = _ANSI_ESCAPE_RE.sub("", self.progress_buffer).strip()
        if not text or text == self.last_progress_text:
            return ""
        now = self.clock()
        if (
            not force
            and self.last_progress_log_time is not None
            and now - self.last_progress_log_time < PROGRESS_LOG_INTERVAL_SECONDS
        ):
            return ""
        self.last_progress_text = text
        self.last_progress_log_time = now
        return text + "\n"

    def flush_pending(self) -> None:
        with self.lock:
            if self.progress_buffer is not None:
                progress = self._progress_line(force=True)
                if progress:
                    self.log_file.write(progress)
                self.progress_buffer = None
            for _, last_written, latest in self.line_progress.values():
                if latest != last_written:
                    self.log_file.write(latest + "\n")
            self.line_progress.clear()
            self.log_file.flush()

    def flush(self) -> None:
        with self.lock:
            self.console.flush()
            self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.console, name)


def _logging_handlers() -> Iterator[logging.StreamHandler]:
    seen: set[int] = set()
    loggers = [logging.getLogger()]
    loggers.extend(
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    for logger in loggers:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and id(handler) not in seen:
                seen.add(id(handler))
                yield handler


@contextmanager
def tee_job_output(log_path: Path) -> Iterator[None]:
    """Send Python stdout, stderr, and logging to the console and runner.log."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_showwarning = warnings.showwarning
    seen_warnings: set[tuple[str, type[Warning], str, int]] = set()

    def show_warning_once(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        key = (str(message), category, filename, lineno)
        if key in seen_warnings:
            return
        seen_warnings.add(key)
        original_showwarning(message, category, filename, lineno, file=file, line=line)

    lock = threading.Lock()
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        stdout = _TeeStream(original_stdout, log_file, lock)
        stderr = _TeeStream(original_stderr, log_file, lock)
        changed_handlers: list[tuple[logging.StreamHandler, Any]] = []
        for handler in _logging_handlers():
            if handler.stream is original_stdout:
                changed_handlers.append((handler, original_stdout))
                handler.setStream(stdout)
            elif handler.stream is original_stderr:
                changed_handlers.append((handler, original_stderr))
                handler.setStream(stderr)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("once")
                warnings.showwarning = show_warning_once
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    yield
        finally:
            stdout.flush_pending()
            stderr.flush_pending()
            for handler, original_stream in reversed(changed_handlers):
                handler.setStream(original_stream)


def run_logged_command(command: list[str], **kwargs: Any) -> None:
    """Run a subprocess and forward its combined output through the active tee."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        **kwargs,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
