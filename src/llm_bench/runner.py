"""Run llama-bench as a subprocess with real-time stderr streaming."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Callable


def get_llama_bench_version(llama_bench: str) -> str:
    """Return the build version string from llama-bench, or 'unknown'."""
    try:
        result = subprocess.run(
            [llama_bench, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        line = (result.stdout or result.stderr).strip().splitlines()
        return line[0] if line else "unknown"
    except Exception:
        pass
    # Fallback: parse from a --help run that prints build info
    try:
        result = subprocess.run(
            [llama_bench, "-h"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in (result.stdout + result.stderr).splitlines():
            if "build:" in line.lower():
                return line.strip()
    except Exception:
        pass
    return "unknown"


def run_benchmark(
    llama_bench: str,
    hf_repo: str,
    n_prompt: int,
    n_gen: int,
    repetitions: int,
    hf_token: str | None,
    extra_args: list[str],
    on_status: Callable[[str], None],
) -> tuple[str, str, int]:
    """
    Run llama-bench for one model.

    Streams stderr line-by-line through on_status so callers can show live
    download/benchmark progress. Returns (stdout, stderr, returncode).
    """
    if not Path(llama_bench).exists():
        raise FileNotFoundError(f"llama-bench not found: {llama_bench}")

    cmd = [
        llama_bench,
        "-hf", hf_repo,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", str(repetitions),
        "-o", "json",
    ]
    if hf_token:
        cmd.extend(["-hft", hf_token])
    cmd.extend(extra_args)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stderr_lines: list[str] = []

    def _drain() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stripped = line.rstrip()
            stderr_lines.append(stripped)
            if stripped:
                on_status(stripped)

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()

    assert process.stdout is not None
    stdout_data = process.stdout.read()
    process.wait()
    drain_thread.join(timeout=5)

    return stdout_data, "\n".join(stderr_lines), process.returncode
