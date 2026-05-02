"""Tests for llm_bench.sysinfo."""

from llm_bench.sysinfo import SystemInfo, collect


def test_collect_returns_sysinfo():
    info = collect()
    assert isinstance(info, SystemInfo)
    assert info.cpu_cores > 0
    assert info.total_ram_gb > 0
    assert info.available_ram_gb > 0
    assert info.available_ram_gb <= info.total_ram_gb


def test_fingerprint_is_stable():
    info = collect()
    assert len(info.fingerprint) == 16
    assert info.fingerprint == info.fingerprint


def test_fingerprint_changes_with_ram():
    info_a = SystemInfo(
        cpu_model="Intel i7", cpu_cores=8, total_ram_gb=16.0,
        available_ram_gb=8.0, os="Linux", gpus=[]
    )
    info_b = SystemInfo(
        cpu_model="Intel i7", cpu_cores=8, total_ram_gb=32.0,
        available_ram_gb=8.0, os="Linux", gpus=[]
    )
    assert info_a.fingerprint != info_b.fingerprint


def test_has_gpu_false_when_no_gpus():
    info = SystemInfo(
        cpu_model="Intel i7", cpu_cores=8, total_ram_gb=16.0,
        available_ram_gb=8.0, os="Linux", gpus=[]
    )
    assert not info.has_gpu
