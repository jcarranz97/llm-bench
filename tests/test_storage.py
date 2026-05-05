"""Tests for llm_bench.storage (uses tmp_path, never touches ~/.llm-bench)."""

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


def test_find_cached_result_miss_on_hw_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_default_backend_keeps_old_cache_key() -> None:
    """Adding 'backend' field with default value must NOT change existing cache keys."""
    meta = _make_meta()
    # The old formula was sha256(hw:n_prompt:n_gen:rep:version)[:16].
    # When backend == 'llama-bench' (default) and label/server_url are unset,
    # the new formula must produce the same hash.
    import hashlib

    expected_raw = (
        f"{meta.hw_fingerprint}:{meta.n_prompt}:{meta.n_gen}"
        f":{meta.repetitions}:{meta.llama_bench_version}"
    )
    expected = hashlib.sha256(expected_raw.encode()).hexdigest()[:16]
    assert meta.config_fingerprint() == expected


def test_backend_changes_cache_key() -> None:
    """Switching backend to 'lm-studio' must produce a different cache key."""
    base = _make_meta()
    lm = RunMeta(**{**base.__dict__, "backend": "lm-studio"})
    assert base.config_fingerprint() != lm.config_fingerprint()


def test_label_isolates_two_machines() -> None:
    """Two LM Studio runs that differ only in `label` must NOT collide."""
    a = _make_meta()
    a.backend = "lm-studio"
    a.label = "desktop"
    a.server_url = "http://localhost:1234"

    b = RunMeta(**{**a.__dict__, "label": "homelab"})
    assert a.model_cache_key("qwen/qwen3-4b") != b.model_cache_key("qwen/qwen3-4b")


def test_old_meta_json_loads_without_new_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run saved before backend/server_url/label existed must still load."""
    import json as _json

    monkeypatch.setattr("llm_bench.storage._RESULTS_DIR", tmp_path / "results")
    run_dir = tmp_path / "results" / "old-run"
    run_dir.mkdir(parents=True)
    legacy_meta = {
        "run_id": "old-run",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "hw_fingerprint": "abc123",
        "llama_bench_version": "build 9000",
        "n_prompt": 512,
        "n_gen": 200,
        "repetitions": 5,
        "profile_name": "medium_ram",
        "model_count": 1,
    }
    (run_dir / "meta.json").write_text(_json.dumps(legacy_meta))
    (run_dir / "results.json").write_text("[]")

    from llm_bench.storage import load_run

    meta, results, _cached = load_run("old-run")
    assert meta.run_id == "old-run"
    assert meta.backend == "llama-bench"  # default fills in
    assert meta.label is None
    assert results == []
