"""Run llama-bench as a subprocess with real-time stderr streaming."""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable, Mapping
from pathlib import Path


def _build_env(env_vars: Mapping[str, str] | None) -> dict[str, str] | None:
    """Layer overrides on top of the parent env. Returns None when no overrides."""
    if not env_vars:
        return None
    return {**os.environ, **env_vars}


def get_llama_bench_version(llama_bench: str, env_vars: Mapping[str, str] | None = None) -> str:
    """Return the build version string from llama-bench, or 'unknown'."""
    env = _build_env(env_vars)
    try:
        result = subprocess.run(
            [llama_bench, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
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
            env=env,
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
    env_vars: Mapping[str, str] | None = None,
    local_path: str = "",
) -> tuple[str, str, int]:
    """
    Run llama-bench for one model.

    Streams stderr line-by-line through on_status so callers can show live
    download/benchmark progress. Returns (stdout, stderr, returncode).

    Pass exactly one of `hf_repo` (downloads via `-hf`) or `local_path`
    (loads a local GGUF via `-m`).

    `env_vars` is layered on top of the parent environment — useful for things
    like `HIP_VISIBLE_DEVICES=0` that need to be set before the binary launches.
    """
    if not Path(llama_bench).exists():
        raise FileNotFoundError(f"llama-bench not found: {llama_bench}")
    if bool(hf_repo) == bool(local_path):
        raise ValueError("run_benchmark requires exactly one of hf_repo or local_path")

    cmd = [llama_bench]
    if local_path:
        cmd.extend(["-m", local_path])
    else:
        cmd.extend(["-hf", hf_repo])
    cmd.extend(
        [
            "-p",
            str(n_prompt),
            "-n",
            str(n_gen),
            "-r",
            str(repetitions),
            "-o",
            "json",
        ]
    )
    if hf_token and hf_repo:
        cmd.extend(["-hft", hf_token])
    cmd.extend(extra_args)

    # Use binary mode so we can split on \r AND \n.
    # llama-bench uses \r for in-place progress bars which Python's text-mode
    # line iterator never yields until a \n appears.
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_build_env(env_vars),
    )

    stderr_lines: list[str] = []

    def _drain() -> None:
        assert process.stderr is not None
        buf = bytearray()
        while True:
            ch = process.stderr.read(1)
            if not ch:
                break
            if ch in (b"\r", b"\n"):
                line = buf.decode("utf-8", errors="replace").strip()
                buf.clear()
                if line:
                    stderr_lines.append(line)
                    on_status(line)
            else:
                buf += ch
        if buf:
            line = buf.decode("utf-8", errors="replace").strip()
            if line:
                stderr_lines.append(line)
                on_status(line)

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()

    assert process.stdout is not None
    stdout_bytes = process.stdout.read()
    stdout_data = stdout_bytes.decode("utf-8", errors="replace")
    process.wait()
    drain_thread.join(timeout=5)

    return stdout_data, "\n".join(stderr_lines), process.returncode
