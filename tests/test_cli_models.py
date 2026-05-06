"""Tests for CLI helper that parses --models for the llama-bench backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_bench.cli import _llama_bench_models_from_csv  # pyright: ignore[reportPrivateUsage]


def test_csv_detects_existing_local_gguf(tmp_path: Path) -> None:
    gguf = tmp_path / "Qwen2.5-7B-Q4.gguf"
    gguf.write_bytes(b"\x00")
    out = _llama_bench_models_from_csv(str(gguf))
    assert len(out) == 1
    assert out[0].local_path == str(gguf)
    assert out[0].hf_repo == ""
    assert out[0].name == "Qwen2.5-7B-Q4.gguf"


def test_csv_detects_gguf_suffix_even_when_path_missing(tmp_path: Path) -> None:
    """A user can point at a path that doesn't exist yet — we still treat it as local."""
    missing = tmp_path / "does-not-exist.gguf"
    out = _llama_bench_models_from_csv(str(missing))
    assert out[0].local_path == str(missing)
    assert out[0].hf_repo == ""


def test_csv_treats_org_repo_as_hf() -> None:
    out = _llama_bench_models_from_csv("ggml-org/gemma-3-1b-it-GGUF")
    assert len(out) == 1
    assert out[0].hf_repo == "ggml-org/gemma-3-1b-it-GGUF"
    assert out[0].local_path == ""
    assert out[0].identifier == "ggml-org/gemma-3-1b-it-GGUF"


def test_csv_mixed_locals_and_repos(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00")
    out = _llama_bench_models_from_csv(f"{gguf},ggml-org/gemma-3-1b-it-GGUF")
    assert len(out) == 2
    assert out[0].local_path == str(gguf)
    assert out[1].hf_repo == "ggml-org/gemma-3-1b-it-GGUF"


def test_csv_strips_whitespace_and_skips_empty() -> None:
    out = _llama_bench_models_from_csv("  ggml-org/foo  , , ggml-org/bar ")
    assert [m.hf_repo for m in out] == ["ggml-org/foo", "ggml-org/bar"]


def test_csv_expands_user_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`~` in paths must be expanded so cache keys / -m args don't contain literal `~`."""
    monkeypatch.setenv("HOME", str(tmp_path))
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00")
    out = _llama_bench_models_from_csv("~/model.gguf")
    assert out[0].local_path == str(gguf)
    assert "~" not in out[0].local_path
