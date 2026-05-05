"""Tests for the llama-bench subprocess runner — focused on env handling."""

from __future__ import annotations

import os

from llm_bench.runner import _build_env


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
