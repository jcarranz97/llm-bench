"""Tests for llm_bench.lm_studio_runner using a fake client."""

from __future__ import annotations

from typing import Any

import pytest

from llm_bench.lm_studio import LMStudioClient, LMStudioError
from llm_bench.lm_studio_runner import run_lm_studio_benchmark


class _FakeClient(LMStudioClient):
    """Minimal stand-in: queue per-call responses; raise on overflow."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        # Skip parent __init__; we don't open sockets.
        self.base_url = "http://test"
        self.timeout = 5.0
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        model: str,
        system: str,
        user_input: str,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "user_input": user_input,
                "max_output_tokens": max_output_tokens,
            }
        )
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _stats(input_tokens: int = 100, ttft: float = 0.5, tps: float = 30.0) -> dict[str, Any]:
    return {
        "stats": {
            "input_tokens": input_tokens,
            "time_to_first_token_seconds": ttft,
            "tokens_per_second": tps,
            "total_output_tokens": 50,
        }
    }


def test_pp_tg_probe_computes_pp_from_ttft_and_tg_from_tps() -> None:
    # 2 reps: pp probes (input=200, ttft=0.5 → 400 t/s ; input=200, ttft=0.4 → 500 t/s)
    # then 2 reps: tg probes (tps=30 ; tps=32)
    client = _FakeClient(
        [
            _stats(input_tokens=200, ttft=0.5),
            _stats(input_tokens=200, ttft=0.4),
            _stats(tps=30.0),
            _stats(tps=32.0),
        ]
    )
    result = run_lm_studio_benchmark(
        client=client,
        model_id="qwen/qwen3-4b",
        n_prompt=128,
        n_gen=64,
        repetitions=2,
        probe="pp-tg",
        on_status=lambda s: None,
    )
    assert result.error is None
    assert result.pp_avg_ts == pytest.approx((400.0 + 500.0) / 2)
    assert result.pp_std_ts is not None and result.pp_std_ts > 0
    assert result.tg_avg_ts == pytest.approx((30.0 + 32.0) / 2)
    assert result.backend == "lm-studio"
    # First two calls are pp probes — max_output_tokens > 1 is required because
    # LM Studio reports TTFT as 0 when only one token is generated.
    assert client.calls[0]["max_output_tokens"] > 1
    assert client.calls[1]["max_output_tokens"] == client.calls[0]["max_output_tokens"]
    # Last two calls are tg probes (max_output_tokens=n_gen)
    assert client.calls[2]["max_output_tokens"] == 64
    assert client.calls[3]["max_output_tokens"] == 64


def test_single_probe_only_fills_tg() -> None:
    client = _FakeClient([_stats(tps=25.0), _stats(tps=27.0), _stats(tps=26.0)])
    result = run_lm_studio_benchmark(
        client=client,
        model_id="m",
        n_prompt=128,
        n_gen=64,
        repetitions=3,
        probe="single",
        on_status=lambda s: None,
    )
    assert result.error is None
    assert result.pp_avg_ts is None
    assert result.tg_avg_ts == pytest.approx((25.0 + 27.0 + 26.0) / 3)
    assert all(c["max_output_tokens"] is None for c in client.calls)


def test_pp_tg_probe_records_error_on_http_failure() -> None:
    client = _FakeClient([LMStudioError("boom"), _stats()])
    result = run_lm_studio_benchmark(
        client=client,
        model_id="m",
        n_prompt=128,
        n_gen=64,
        repetitions=1,
        probe="pp-tg",
        on_status=lambda s: None,
    )
    assert result.error is not None
    assert "pp probe failed" in result.error
    # tg probe should NOT have been attempted
    assert len(client.calls) == 1


def test_unknown_probe_returns_error() -> None:
    client = _FakeClient([])
    result = run_lm_studio_benchmark(
        client=client,
        model_id="m",
        n_prompt=128,
        n_gen=64,
        repetitions=1,
        probe="bogus",
        on_status=lambda s: None,
    )
    assert result.error is not None
    assert "unknown probe" in result.error
    assert client.calls == []


def test_single_probe_without_tps_reports_error() -> None:
    client = _FakeClient([{"stats": {"tokens_per_second": 0}}])
    result = run_lm_studio_benchmark(
        client=client,
        model_id="m",
        n_prompt=128,
        n_gen=64,
        repetitions=1,
        probe="single",
        on_status=lambda s: None,
    )
    assert result.error is not None
    assert "tokens_per_second" in result.error
