"""Tests for llm_bench.parser."""

import json

from llm_bench.parser import BenchResult, extract_json, parse_bench_output

_SAMPLE_ENTRIES = [
    {
        "model_type": "gemma4 E2B Q4_K - Medium",
        "model_size": 1_610_612_736,
        "model_n_params": 2_000_000_000,
        "backends": ["CPU"],
        "n_threads": 4,
        "n_prompt": 512,
        "n_gen": 0,
        "avg_ts": 55.3,
        "stddev_ts": 0.5,
    },
    {
        "model_type": "gemma4 E2B Q4_K - Medium",
        "model_size": 1_610_612_736,
        "model_n_params": 2_000_000_000,
        "backends": ["CPU"],
        "n_threads": 4,
        "n_prompt": 0,
        "n_gen": 200,
        "avg_ts": 12.8,
        "stddev_ts": 0.1,
    },
]


def test_extract_json_clean():
    raw = json.dumps(_SAMPLE_ENTRIES)
    result = extract_json(raw)
    assert len(result) == 2


def test_extract_json_with_noise():
    raw = "Downloading model.gguf ── 100%\n" + json.dumps(_SAMPLE_ENTRIES)
    result = extract_json(raw)
    assert len(result) == 2


def test_extract_json_empty():
    assert extract_json("") == []
    assert extract_json("no json here") == []


def test_parse_bench_output_basic():
    result = parse_bench_output("My Model", "org/model", _SAMPLE_ENTRIES)
    assert result.error is None
    assert result.model_type == "gemma4 E2B Q4_K - Medium"
    assert result.pp_avg_ts == 55.3
    assert result.tg_avg_ts == 12.8
    assert result.backend == "CPU"
    assert result.threads == 4


def test_parse_bench_output_empty():
    result = parse_bench_output("My Model", "org/model", [])
    assert result.error == "No data returned"
    assert result.pp_avg_ts is None
    assert result.tg_avg_ts is None


def test_bench_result_roundtrip():
    r = BenchResult(
        model_name="Test",
        hf_repo="org/test",
        pp_avg_ts=50.0,
        tg_avg_ts=10.0,
    )
    d = r.to_dict()
    r2 = BenchResult.from_dict(d)
    assert r2.model_name == r.model_name
    assert r2.pp_avg_ts == r.pp_avg_ts


def test_model_size_properties():
    r = BenchResult(
        model_name="Test",
        hf_repo="org/test",
        model_size_bytes=5_316_915_200,  # ~4.95 GiB
        model_n_params=7_520_000_000,
    )
    assert abs(r.model_size_gib - 4.95) < 0.01
    assert abs(r.model_params_b - 7.52) < 0.01
