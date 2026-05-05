"""Tests for llm_bench.models YAML loader and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_bench.models import Model, get_profile_by_name, load_profile_from_file


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content)


def test_model_identifier_prefers_lm_studio_id() -> None:
    m = Model(name="x", lm_studio_id="qwen/qwen3-4b", hf_repo="some/repo")
    assert m.identifier == "qwen/qwen3-4b"


def test_model_identifier_falls_back_to_hf_repo() -> None:
    m = Model(name="x", hf_repo="org/repo")
    assert m.identifier == "org/repo"


def test_loader_accepts_lm_studio_id_only(tmp_path: Path) -> None:
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(
        yaml_path,
        """
profile: test
description: x
min_ram_gb: 0
max_ram_gb: 9999
models:
  - name: Foo
    lm_studio_id: foo/bar
""",
    )
    profile = load_profile_from_file(yaml_path)
    assert len(profile.models) == 1
    assert profile.models[0].lm_studio_id == "foo/bar"
    assert profile.models[0].hf_repo == ""
    assert profile.models[0].identifier == "foo/bar"


def test_loader_rejects_entry_without_any_id(tmp_path: Path) -> None:
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(
        yaml_path,
        """
profile: test
description: x
min_ram_gb: 0
max_ram_gb: 9999
models:
  - name: NoId
""",
    )
    with pytest.raises(ValueError, match="hf_repo / lm_studio_id / local_path"):
        load_profile_from_file(yaml_path)


def test_loader_accepts_extra_args_as_list(tmp_path: Path) -> None:
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(
        yaml_path,
        """
profile: test
description: x
min_ram_gb: 0
max_ram_gb: 9999
models:
  - name: BigMoE
    hf_repo: org/big-moe-gguf
    extra_args: ["-ngl", "30", "-fa", "1"]
""",
    )
    profile = load_profile_from_file(yaml_path)
    assert profile.models[0].extra_args == ["-ngl", "30", "-fa", "1"]


def test_loader_accepts_extra_args_as_string(tmp_path: Path) -> None:
    """A single string is shlex-split — convenient for short YAML."""
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(
        yaml_path,
        """
profile: test
description: x
min_ram_gb: 0
max_ram_gb: 9999
models:
  - name: Tuned
    hf_repo: org/tuned
    extra_args: "-ngl 30 -fa 1"
""",
    )
    profile = load_profile_from_file(yaml_path)
    assert profile.models[0].extra_args == ["-ngl", "30", "-fa", "1"]


def test_loader_extra_args_default_empty(tmp_path: Path) -> None:
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(
        yaml_path,
        """
profile: test
description: x
min_ram_gb: 0
max_ram_gb: 9999
models:
  - name: Plain
    hf_repo: org/plain
""",
    )
    profile = load_profile_from_file(yaml_path)
    assert profile.models[0].extra_args == []


def test_loader_extra_args_rejects_non_list_non_string(tmp_path: Path) -> None:
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(
        yaml_path,
        """
profile: test
description: x
min_ram_gb: 0
max_ram_gb: 9999
models:
  - name: Bad
    hf_repo: org/bad
    extra_args: 42
""",
    )
    with pytest.raises(ValueError, match="extra_args must be a list or string"):
        load_profile_from_file(yaml_path)


def test_bundled_lm_studio_profile_loads() -> None:
    """The bundled lm_studio.yaml ships and is discoverable by name."""
    profile = get_profile_by_name("lm_studio")
    assert profile is not None
    assert profile.profile == "lm_studio"
    assert len(profile.models) > 0
    # Every entry should be addressable as an LM Studio model.
    for m in profile.models:
        assert m.lm_studio_id, f"model '{m.name}' missing lm_studio_id"
        assert m.identifier == m.lm_studio_id
