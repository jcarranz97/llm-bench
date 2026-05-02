"""Tests for llm_bench.storage (uses tmp_path, never touches ~/.llm-bench)."""

import json
from pathlib import Path

import pytest

from llm_bench.parser import BenchResult
from llm_bench.storage import RunMeta, find_cached_result, load_run, save_run


def _make_meta(run_id: str = "20240101-000000-abc123") -> RunMeta:
    return RunMeta(
        run_id=run_id,
        timestamp="2024-01-01T00:00:00+00:00",
        hw_fingerprint="deadbeef12345678",
        llama_bench_version="build 9009",
        n_prompt=512,
        n_gen=200,
        repetitions=5,
        profile_name="medium_ram",
        model_count=3,
    )


def _make_result(name: str = "Model A", hf_repo: str = "org/model-a") -> BenchResult:
    return BenchResult(
        model_name=name,
        hf_repo=hf_repo,
        model_type="test 7B Q4_K",
        pp_avg_ts=50.0,
        pp_std_ts=0.5,
        tg_avg_ts=12.0,
        tg_std_ts=0.1,
        backend="CPU",
        threads=4,
    )


def test_save_and_load_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_bench.storage._RESULTS_DIR", tmp_path / "results")

    meta = _make_meta()
    results = [_make_result()]
    cached = {"org/model-a": False}

    run_dir = save_run(meta, results, cached)
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "results.json").exists()

    loaded_meta, loaded_results, loaded_cached = load_run(meta.run_id)
    assert loaded_meta.run_id == meta.run_id
    assert len(loaded_results) == 1
    assert loaded_results[0].hf_repo == "org/model-a"
    assert not loaded_cached["org/model-a"]


def test_cache_key_stable(tmp_path: Path) -> None:
    meta = _make_meta()
    key1 = meta.model_cache_key("org/model-a")
    key2 = meta.model_cache_key("org/model-a")
    assert key1 == key2
    assert key1 != meta.model_cache_key("org/model-b")


def test_find_cached_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_bench.storage._RESULTS_DIR", tmp_path / "results")

    meta = _make_meta()
    r = _make_result()
    save_run(meta, [r], {"org/model-a": False})

    found = find_cached_result(meta, "org/model-a")
    assert found is not None
    assert found.hf_repo == "org/model-a"
    assert found.tg_avg_ts == 12.0


def test_find_cached_result_miss_on_hw_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_bench.storage._RESULTS_DIR", tmp_path / "results")

    meta_old = _make_meta()
    save_run(meta_old, [_make_result()], {})

    # Different hardware fingerprint
    meta_new = RunMeta(
        run_id="20240102-000000-ffffff",
        timestamp="2024-01-02T00:00:00+00:00",
        hw_fingerprint="ffffffff00000000",  # changed
        llama_bench_version="build 9009",
        n_prompt=512,
        n_gen=200,
        repetitions=5,
        profile_name="medium_ram",
        model_count=3,
    )
    found = find_cached_result(meta_new, "org/model-a")
    assert found is None
