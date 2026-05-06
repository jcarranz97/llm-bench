"""Tests for llm_bench.models YAML loader and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_bench.models import (
    GpuMatch,
    Model,
    ModelProfile,
    get_profile_by_name,
    gpu_matches_profile,
    load_profile_from_file,
    matching_gpu_profiles,
    select_profile,
)
from llm_bench.sysinfo import GpuInfo, SystemInfo


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


# ── GPU-specific profile tests ────────────────────────────────────────────────


def _make_sysinfo(gpus: list[GpuInfo], ram_gb: float = 32.0) -> SystemInfo:
    return SystemInfo(
        cpu_model="test-cpu",
        cpu_cores=8,
        total_ram_gb=ram_gb,
        available_ram_gb=ram_gb * 0.8,
        os="linux",
        gpus=gpus,
    )


def test_loader_parses_gpu_match(tmp_path: Path) -> None:
    yaml_path = tmp_path / "specific" / "amd" / "RX7900XT.yaml"
    yaml_path.parent.mkdir(parents=True)
    _write_yaml(
        yaml_path,
        """
profile: gpu_amd_rx7900xt
description: x
min_ram_gb: 16
max_ram_gb: 9999
gpu_match:
  vendor: amd
  name_contains: ["RX 7900 XT", "Navi 31"]
  backends: [ROCm, Vulkan]
  min_vram_gb: 18
models:
  - name: Foo
    hf_repo: foo/bar
""",
    )
    profile = load_profile_from_file(yaml_path)
    assert profile.gpu_match is not None
    assert profile.gpu_match.vendor == "amd"
    assert profile.gpu_match.name_contains == ["RX 7900 XT", "Navi 31"]
    assert profile.gpu_match.backends == ["ROCm", "Vulkan"]
    assert profile.gpu_match.min_vram_gb == 18.0


def test_loader_rejects_invalid_vendor(tmp_path: Path) -> None:
    yaml_path = tmp_path / "p.yaml"
    _write_yaml(
        yaml_path,
        """
profile: bogus
description: x
min_ram_gb: 0
max_ram_gb: 9999
gpu_match:
  vendor: bogusco
models:
  - name: Foo
    hf_repo: foo/bar
""",
    )
    with pytest.raises(ValueError, match="gpu_match.vendor"):
        load_profile_from_file(yaml_path)


def test_loader_rejects_vendor_path_mismatch(tmp_path: Path) -> None:
    yaml_path = tmp_path / "specific" / "amd" / "X.yaml"
    yaml_path.parent.mkdir(parents=True)
    _write_yaml(
        yaml_path,
        """
profile: gpu_nv_x
description: x
min_ram_gb: 0
max_ram_gb: 9999
gpu_match:
  vendor: nvidia
models:
  - name: Foo
    hf_repo: foo/bar
""",
    )
    with pytest.raises(ValueError, match="does not match"):
        load_profile_from_file(yaml_path)


def test_gpu_matches_profile_substring_case_insensitive() -> None:
    match = GpuMatch(vendor="amd", name_contains=["RX 7900 XT"], backends=["ROCm"])
    rocm_card = GpuInfo(name="AMD Radeon RX 7900 XT", vram_gb=20.0, backend="ROCm")
    other_card = GpuInfo(name="AMD Radeon RX 6800", vram_gb=16.0, backend="ROCm")
    assert gpu_matches_profile(match, rocm_card)
    assert not gpu_matches_profile(match, other_card)


def test_gpu_matches_profile_uses_default_backends() -> None:
    """When YAML omits `backends:`, the vendor's default set applies."""
    match = GpuMatch(vendor="amd", name_contains=["RX 7900 XT"])
    rocm_card = GpuInfo(name="Radeon RX 7900 XT", vram_gb=20.0, backend="ROCm")
    cuda_card = GpuInfo(name="Radeon RX 7900 XT impostor", vram_gb=20.0, backend="CUDA")
    assert gpu_matches_profile(match, rocm_card)
    assert not gpu_matches_profile(match, cuda_card)


def test_gpu_matches_profile_respects_min_vram() -> None:
    match = GpuMatch(vendor="amd", name_contains=["RX 7900 XT"], min_vram_gb=18.0)
    big = GpuInfo(name="RX 7900 XT", vram_gb=20.0, backend="ROCm")
    small = GpuInfo(name="RX 7900 XT", vram_gb=12.0, backend="ROCm")
    unknown = GpuInfo(name="RX 7900 XT", vram_gb=None, backend="ROCm")
    assert gpu_matches_profile(match, big)
    assert not gpu_matches_profile(match, small)
    # Unknown VRAM is treated as "allow" so partial detection (lspci-only) doesn't lock users out.
    assert gpu_matches_profile(match, unknown)


def test_select_profile_excludes_gpu_specific(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """select_profile() must never auto-pick a profile that has gpu_match set."""
    yaml_path = tmp_path / "models" / "specific" / "amd" / "card.yaml"
    yaml_path.parent.mkdir(parents=True)
    _write_yaml(
        yaml_path,
        """
profile: gpu_amd_card
description: x
min_ram_gb: 0
max_ram_gb: 9999
gpu_match:
  vendor: amd
  name_contains: ["foo"]
models:
  - name: Foo
    hf_repo: foo/bar
""",
    )
    monkeypatch.setattr("llm_bench.models._USER_MODELS_DIR", tmp_path / "models")
    info = _make_sysinfo([], ram_gb=32.0)
    chosen = select_profile(info)
    assert chosen.gpu_match is None
    assert chosen.profile != "gpu_amd_card"


def test_matching_gpu_profiles_filters_by_physical_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """matching_gpu_profiles returns only profiles whose gpu_match fits a physical GPU."""
    amd_yaml = tmp_path / "specific" / "amd" / "card.yaml"
    nv_yaml = tmp_path / "specific" / "nvidia" / "card.yaml"
    amd_yaml.parent.mkdir(parents=True)
    nv_yaml.parent.mkdir(parents=True)
    _write_yaml(
        amd_yaml,
        """
profile: gpu_amd_test
description: x
min_ram_gb: 0
max_ram_gb: 9999
gpu_match:
  vendor: amd
  name_contains: ["RX 7900 XT"]
  backends: [ROCm]
models:
  - name: Foo
    hf_repo: foo/bar
""",
    )
    _write_yaml(
        nv_yaml,
        """
profile: gpu_nvidia_test
description: x
min_ram_gb: 0
max_ram_gb: 9999
gpu_match:
  vendor: nvidia
  name_contains: ["RTX 4090"]
models:
  - name: Bar
    hf_repo: bar/baz
""",
    )
    monkeypatch.setattr("llm_bench.models._USER_MODELS_DIR", tmp_path)

    info = _make_sysinfo([GpuInfo(name="AMD Radeon RX 7900 XT", vram_gb=20.0, backend="ROCm")])
    matches = matching_gpu_profiles(info, runtime_devices=None)
    names = {p.profile for p in matches}
    assert "gpu_amd_test" in names
    assert "gpu_nvidia_test" not in names


def test_matching_gpu_profiles_warn_and_proceed_when_devices_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """runtime_devices=None means the probe couldn't run — match should still apply."""
    yaml_path = tmp_path / "specific" / "amd" / "card.yaml"
    yaml_path.parent.mkdir(parents=True)
    _write_yaml(
        yaml_path,
        """
profile: gpu_amd_proceed
description: x
min_ram_gb: 0
max_ram_gb: 9999
gpu_match:
  vendor: amd
  name_contains: ["RX 7900 XT"]
  backends: [ROCm]
models:
  - name: Foo
    hf_repo: foo/bar
""",
    )
    monkeypatch.setattr("llm_bench.models._USER_MODELS_DIR", tmp_path)
    info = _make_sysinfo([GpuInfo(name="RX 7900 XT", vram_gb=20.0, backend="ROCm")])
    assert any(p.profile == "gpu_amd_proceed" for p in matching_gpu_profiles(info, None))


def test_matching_gpu_profiles_skips_when_runtime_doesnt_see_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty runtime_devices list (probe ran but found nothing) blocks the match."""
    from llm_bench.devices import RuntimeDevice

    yaml_path = tmp_path / "specific" / "amd" / "card.yaml"
    yaml_path.parent.mkdir(parents=True)
    _write_yaml(
        yaml_path,
        """
profile: gpu_amd_blocked
description: x
min_ram_gb: 0
max_ram_gb: 9999
gpu_match:
  vendor: amd
  name_contains: ["RX 7900 XT"]
  backends: [ROCm]
models:
  - name: Foo
    hf_repo: foo/bar
""",
    )
    monkeypatch.setattr("llm_bench.models._USER_MODELS_DIR", tmp_path)
    info = _make_sysinfo([GpuInfo(name="RX 7900 XT", vram_gb=20.0, backend="ROCm")])

    # Empty list: probe ran but found no matching device → matcher must not select it.
    assert matching_gpu_profiles(info, runtime_devices=[]) == []

    # CPU-only runtime device: not the AMD GPU → no match.
    cpu_only = [RuntimeDevice(name="CPU", description="Generic CPU", backend="CPU")]
    assert matching_gpu_profiles(info, runtime_devices=cpu_only) == []

    # ROCm runtime device with the card name → match.
    ok = [RuntimeDevice(name="ROCm0", description="AMD Radeon RX 7900 XT", backend="ROCm")]
    matches = matching_gpu_profiles(info, runtime_devices=ok)
    assert any(p.profile == "gpu_amd_blocked" for p in matches)


def test_recursive_loader_finds_subdirectory_yamls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """User profiles placed under specific/<vendor>/ should be discovered."""
    nested = tmp_path / "specific" / "amd" / "RX9999.yaml"
    nested.parent.mkdir(parents=True)
    _write_yaml(
        nested,
        """
profile: gpu_amd_rx9999
description: nested
min_ram_gb: 0
max_ram_gb: 9999
gpu_match:
  vendor: amd
  name_contains: ["RX 9999"]
models:
  - name: Foo
    hf_repo: foo/bar
""",
    )
    monkeypatch.setattr("llm_bench.models._USER_MODELS_DIR", tmp_path)
    assert get_profile_by_name("gpu_amd_rx9999") is not None


def test_bundled_amd_rx7900xt_profile_loads() -> None:
    """The migrated specific/amd/RX7900XT.yaml ships and is well-formed."""
    profile = get_profile_by_name("gpu_amd_rx7900xt")
    assert profile is not None, "gpu_amd_rx7900xt profile should be bundled"
    assert profile.gpu_match is not None
    assert profile.gpu_match.vendor == "amd"
    assert "RX 7900 XT" in profile.gpu_match.name_contains
    # Models ship with -fitt so llama-bench auto-fits the model into VRAM
    # rather than blindly defaulting to -ngl 99 and OOM-ing on the largest entries.
    assert all("-fitt" in m.extra_args for m in profile.models)


def test_make_profile_helper_unused_import_silenced() -> None:
    """Touch ModelProfile so the import isn't reported as unused."""
    p = ModelProfile(
        profile="x",
        description="x",
        min_ram_gb=0,
        max_ram_gb=9999,
        models=[],
    )
    assert p.gpu_match is None


def test_find_matched_runtime_device_returns_specific_card() -> None:
    """When two ROCm devices are present, find the one whose description matches."""
    from llm_bench.devices import RuntimeDevice
    from llm_bench.models import find_matched_runtime_device

    match = GpuMatch(vendor="amd", name_contains=["RX 7900 XT"], backends=["ROCm"])
    devices = [
        RuntimeDevice(name="ROCm0", description="AMD Radeon RX 7900 XT", backend="ROCm"),
        RuntimeDevice(name="ROCm1", description="AMD Ryzen 9 iGPU", backend="ROCm"),
    ]
    matched = find_matched_runtime_device(match, devices)
    assert matched is not None
    assert matched.name == "ROCm0"


def test_find_matched_runtime_device_handles_swapped_enumeration() -> None:
    """Same hardware, opposite enumeration order — must return the dGPU not the iGPU."""
    from llm_bench.devices import RuntimeDevice
    from llm_bench.models import find_matched_runtime_device

    match = GpuMatch(vendor="amd", name_contains=["RX 7900 XT"], backends=["ROCm"])
    devices = [
        RuntimeDevice(name="ROCm0", description="AMD Ryzen 9 iGPU", backend="ROCm"),
        RuntimeDevice(name="ROCm1", description="AMD Radeon RX 7900 XT", backend="ROCm"),
    ]
    matched = find_matched_runtime_device(match, devices)
    assert matched is not None
    assert matched.name == "ROCm1"


def test_find_matched_runtime_device_returns_none_when_probe_unavailable() -> None:
    from llm_bench.models import find_matched_runtime_device

    match = GpuMatch(vendor="amd", name_contains=["RX 7900 XT"])
    assert find_matched_runtime_device(match, runtime_devices=None) is None
