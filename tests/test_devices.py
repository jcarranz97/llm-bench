"""Tests for llm_bench.devices runtime probes."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from llm_bench import devices
from llm_bench.devices import (
    RuntimeDevice,
    llama_bench_devices,
    llama_server_devices,
    lm_studio_runtimes,
)
from llm_bench.sysinfo import GpuInfo, SystemInfo


class _FakeCompleted:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _make_sysinfo(gpus: list[GpuInfo]) -> SystemInfo:
    return SystemInfo(
        cpu_model="cpu",
        cpu_cores=8,
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        os="linux",
        gpus=gpus,
    )


# ── llama-bench --list-devices ───────────────────────────────────────────────


def test_llama_bench_devices_parses_canonical_output(monkeypatch) -> None:
    sample = (
        "Available devices:\n"
        "  ROCm0: AMD Radeon RX 7900 XT (20464 MiB, 20256 MiB free)\n"
        "  CPU: AMD Ryzen 9 7950X (16 cores, 32 threads)\n"
    )

    def fake_run(*args: Any, **kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(stdout=sample, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = llama_bench_devices("/fake/llama-bench")
    assert result is not None
    assert len(result) == 2
    rocm = next(d for d in result if d.name == "ROCm0")
    assert rocm.backend == "ROCm"
    assert "RX 7900 XT" in rocm.description


def test_llama_bench_devices_returns_none_for_old_binary(monkeypatch) -> None:
    """Older builds reject `--list-devices` with non-zero exit."""

    def fake_run(*args: Any, **kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(stderr="error: unknown option", returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert llama_bench_devices("/fake/llama-bench") is None


def test_llama_bench_devices_returns_none_when_subprocess_fails(monkeypatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _FakeCompleted:
        raise FileNotFoundError("missing binary")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert llama_bench_devices("/fake/llama-bench") is None


def test_llama_bench_devices_returns_none_for_empty_output(monkeypatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(stdout="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert llama_bench_devices("/fake/llama-bench") is None


# ── llama-server /props.devices ───────────────────────────────────────────────


class _FakeLlamaServerClient:
    def __init__(self, props_payload: dict) -> None:
        self._payload = props_payload

    def props(self) -> dict:
        return self._payload


def test_llama_server_devices_extracts_from_props() -> None:
    client = _FakeLlamaServerClient(
        {
            "build_info": "abc",
            "devices": [
                {"name": "ROCm0", "description": "AMD Radeon RX 7900 XT", "backend": "ROCm"},
                {"name": "CPU", "description": "AMD Ryzen 9 7950X", "backend": "CPU"},
            ],
        }
    )
    result = llama_server_devices(client)  # type: ignore[arg-type]
    assert result is not None
    assert any(d.backend == "ROCm" and "RX 7900 XT" in d.description for d in result)


def test_llama_server_devices_returns_none_when_devices_field_missing() -> None:
    client = _FakeLlamaServerClient({"build_info": "abc"})
    assert llama_server_devices(client) is None  # type: ignore[arg-type]


def test_llama_server_devices_returns_none_on_props_error(monkeypatch) -> None:
    from llm_bench.llama_server import LlamaServerError

    class _ErrorClient:
        def props(self) -> dict:
            raise LlamaServerError("nope")

    assert llama_server_devices(_ErrorClient()) is None  # type: ignore[arg-type]


# ── lms runtime ls + sysinfo cross-reference ─────────────────────────────────


def test_lm_studio_runtimes_synthesises_devices_from_runtimes_and_gpus(monkeypatch) -> None:
    monkeypatch.setattr(devices, "_lms_runtime_ids", lambda: ["rocm-llama.cpp"])
    info = _make_sysinfo(
        [GpuInfo(name="AMD Radeon RX 7900 XT", vram_gb=20.0, backend="ROCm")]
    )
    result = lm_studio_runtimes(info)
    assert result is not None
    assert len(result) == 1
    assert result[0].backend == "ROCm"
    assert "RX 7900 XT" in result[0].description


def test_lm_studio_runtimes_returns_none_when_lms_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(devices, "_lms_runtime_ids", lambda: None)
    info = _make_sysinfo([GpuInfo(name="x", backend="ROCm")])
    assert lm_studio_runtimes(info) is None


def test_lm_studio_runtimes_returns_empty_when_no_compatible_runtime(monkeypatch) -> None:
    """An ROCm GPU + only CPU runtime installed → empty list (probe ran, found nothing)."""
    monkeypatch.setattr(devices, "_lms_runtime_ids", lambda: ["cpu-llama.cpp"])
    info = _make_sysinfo([GpuInfo(name="AMD GPU", backend="ROCm")])
    assert lm_studio_runtimes(info) == []


def test_lm_studio_runtimes_returns_none_when_no_gpu(monkeypatch) -> None:
    monkeypatch.setattr(devices, "_lms_runtime_ids", lambda: ["rocm-llama.cpp"])
    info = _make_sysinfo([])
    # No GPU + ROCm runtime installed → no matches but the probe did run.
    assert lm_studio_runtimes(info) is None


def test_runtime_device_dataclass_roundtrip() -> None:
    d = RuntimeDevice(name="ROCm0", description="X", backend="ROCm")
    assert d.name == "ROCm0"
    assert d.backend == "ROCm"


def test_lms_runtime_ids_handles_subprocess_error(monkeypatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _FakeCompleted:
        raise FileNotFoundError("lms not installed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert devices._lms_runtime_ids() is None


def test_lms_runtime_ids_parses_output(monkeypatch) -> None:
    sample = """\
NAME                          VERSION
rocm-llama.cpp                1.20.0
cpu-llama.cpp                 1.20.0
"""

    def fake_run(*args: Any, **kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(stdout=sample, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    ids = devices._lms_runtime_ids()
    assert ids is not None
    assert "rocm-llama.cpp" in ids
    assert "cpu-llama.cpp" in ids


def test_pytest_imports_all_symbols() -> None:
    """Sanity import check so flake8/F401 doesn't trip over the test module."""
    assert pytest  # touches the import


# ── device_index / auto_env_for_device ────────────────────────────────────────


def test_device_index_extracts_trailing_int() -> None:
    from llm_bench.devices import device_index

    assert device_index(RuntimeDevice(name="ROCm0", description="x", backend="ROCm")) == 0
    assert device_index(RuntimeDevice(name="ROCm1", description="x", backend="ROCm")) == 1
    assert device_index(RuntimeDevice(name="CUDA12", description="x", backend="CUDA")) == 12


def test_device_index_returns_none_for_nameless_device() -> None:
    from llm_bench.devices import device_index

    assert device_index(RuntimeDevice(name="CPU", description="x", backend="CPU")) is None
    assert device_index(RuntimeDevice(name="Metal", description="x", backend="Metal")) is None


def test_auto_env_for_amd_device_sets_hip_visible() -> None:
    from llm_bench.devices import auto_env_for_device

    dev = RuntimeDevice(name="ROCm1", description="AMD Radeon RX 7900 XT", backend="ROCm")
    env = auto_env_for_device(dev, vendor="amd")
    assert env == {"HIP_VISIBLE_DEVICES": "1"}


def test_auto_env_for_nvidia_device_sets_cuda_visible() -> None:
    from llm_bench.devices import auto_env_for_device

    dev = RuntimeDevice(name="CUDA0", description="RTX 4090", backend="CUDA")
    env = auto_env_for_device(dev, vendor="nvidia")
    assert env == {"CUDA_VISIBLE_DEVICES": "0"}


def test_auto_env_returns_empty_when_no_index() -> None:
    from llm_bench.devices import auto_env_for_device

    dev = RuntimeDevice(name="CPU", description="x", backend="CPU")
    assert auto_env_for_device(dev, vendor="amd") == {}


def test_auto_env_returns_empty_for_apple() -> None:
    """Apple's Metal exposes a single device — nothing to pin."""
    from llm_bench.devices import auto_env_for_device

    dev = RuntimeDevice(name="Metal", description="M3 Max", backend="Metal")
    assert auto_env_for_device(dev, vendor="apple") == {}


# ── llama_bench_devices parses real-world output ─────────────────────────────


def test_llama_bench_devices_skips_ggml_init_noise(monkeypatch) -> None:
    """Real `--list-devices` output includes ggml debug lines we must skip."""
    sample = (
        "ggml_cuda_init: found 2 ROCm devices (Total VRAM: 35814 MiB):\n"
        "  Device 0: AMD Radeon RX 7900 XT, gfx1100 (0x1100), VMM: no, "
        "Wave Size: 32, VRAM: 20464 MiB\n"
        "  Device 1: AMD Ryzen 9 7950X 16-Core Processor, gfx1036 (0x1036), "
        "VMM: no, Wave Size: 32, VRAM: 15350 MiB\n"
        "ggml_backend_cuda_get_available_uma_memory: final available_memory_kb: 29048272\n"
        "Available devices:\n"
        "  ROCm0: AMD Radeon RX 7900 XT (20464 MiB, 20428 MiB free)\n"
        "  ROCm1: AMD Ryzen 9 7950X 16-Core Processor (15350 MiB, 28367 MiB free)\n"
    )

    def fake_run(*args: Any, **kwargs: Any) -> _FakeCompleted:
        return _FakeCompleted(stdout=sample, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = llama_bench_devices("/fake/llama-bench")
    assert result is not None
    # Should pick up exactly the two ROCm devices and not the noisy `Device N:`
    # lines or the `ggml_*` ones.
    rocm_devices = [d for d in result if d.backend == "ROCm"]
    assert len(rocm_devices) == 2
    rx7900 = next(d for d in rocm_devices if "RX 7900 XT" in d.description)
    assert rx7900.name == "ROCm0"
    igpu = next(d for d in rocm_devices if "Ryzen" in d.description)
    assert igpu.name == "ROCm1"
