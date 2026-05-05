"""Tests for the llama-bench subprocess runner — focused on env handling."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_bench.runner import _build_env, run_benchmark


def test_build_env_returns_none_when_no_overrides() -> None:
    """No overrides → return None so Popen inherits the parent env directly."""
    assert _build_env(None) is None
    assert _build_env({}) is None


def test_build_env_layers_overrides_on_parent() -> None:
    """Overrides win, but the rest of os.environ comes through unchanged."""
    env = _build_env({"HIP_VISIBLE_DEVICES": "0"})
    assert env is not None
    assert env["HIP_VISIBLE_DEVICES"] == "0"
    # PATH almost always exists on Linux/macOS dev boxes; if it does, it must survive.
    if "PATH" in os.environ:
        assert env["PATH"] == os.environ["PATH"]


def test_build_env_override_replaces_existing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """An override for a var already in the parent env must replace, not append."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "all")
    env = _build_env({"HIP_VISIBLE_DEVICES": "0"})
    assert env is not None
    assert env["HIP_VISIBLE_DEVICES"] == "0"


def test_run_benchmark_rejects_both_hf_and_local(tmp_path: Path) -> None:
    """Caller must pass exactly one of hf_repo / local_path."""
    fake_bench = tmp_path / "llama-bench"
    fake_bench.write_text("#!/bin/sh\necho '[]'\n")
    fake_bench.chmod(0o755)
    with pytest.raises(ValueError, match="exactly one"):
        run_benchmark(
            llama_bench=str(fake_bench),
            hf_repo="org/foo",
            local_path="/tmp/foo.gguf",
            n_prompt=1,
            n_gen=1,
            repetitions=1,
            hf_token=None,
            extra_args=[],
            on_status=lambda _: None,
        )


def test_run_benchmark_rejects_neither_hf_nor_local(tmp_path: Path) -> None:
    fake_bench = tmp_path / "llama-bench"
    fake_bench.write_text("#!/bin/sh\necho '[]'\n")
    fake_bench.chmod(0o755)
    with pytest.raises(ValueError, match="exactly one"):
        run_benchmark(
            llama_bench=str(fake_bench),
            hf_repo="",
            n_prompt=1,
            n_gen=1,
            repetitions=1,
            hf_token=None,
            extra_args=[],
            on_status=lambda _: None,
        )


def test_run_benchmark_passes_local_path_with_dash_m(tmp_path: Path) -> None:
    """A local_path must produce `-m <path>` and NOT `-hf` (and ignore hf_token)."""
    # Echo all argv so we can inspect what the bench would have been invoked with.
    fake_bench = tmp_path / "llama-bench"
    fake_bench.write_text('#!/bin/sh\nfor a in "$@"; do echo "ARG=$a" >&2; done\necho "[]"\n')
    fake_bench.chmod(0o755)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00")

    captured: list[str] = []

    def on_status(line: str) -> None:
        captured.append(line)

    stdout, stderr, rc = run_benchmark(
        llama_bench=str(fake_bench),
        hf_repo="",
        local_path=str(gguf),
        n_prompt=1,
        n_gen=1,
        repetitions=1,
        hf_token="ignored-when-local",
        extra_args=[],
        on_status=on_status,
    )
    assert rc == 0
    args = "\n".join(captured)
    assert "ARG=-m" in args
    assert f"ARG={gguf}" in args
    assert "ARG=-hf" not in args
    assert "ARG=-hft" not in args


def test_run_benchmark_passes_hf_repo_with_dash_hf(tmp_path: Path) -> None:
    fake_bench = tmp_path / "llama-bench"
    fake_bench.write_text('#!/bin/sh\nfor a in "$@"; do echo "ARG=$a" >&2; done\necho "[]"\n')
    fake_bench.chmod(0o755)

    captured: list[str] = []
    stdout, _, rc = run_benchmark(
        llama_bench=str(fake_bench),
        hf_repo="org/some-model",
        n_prompt=1,
        n_gen=1,
        repetitions=1,
        hf_token="abc",
        extra_args=[],
        on_status=captured.append,
    )
    assert rc == 0
    args = "\n".join(captured)
    assert "ARG=-hf" in args
    assert "ARG=org/some-model" in args
    assert "ARG=-hft" in args
    assert "ARG=abc" in args
    assert "ARG=-m" not in args
