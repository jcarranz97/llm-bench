"""Tests for llm_bench.llama_server_runner using a fake client."""

from __future__ import annotations

from typing import Any

import pytest

from llm_bench.llama_server import LlamaServerClient, LlamaServerError
from llm_bench.llama_server_runner import run_llama_server_benchmark


class _FakeClient(LlamaServerClient):
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        # Skip parent __init__; we don't open sockets.
        self.base_url = "http://test"
        self.timeout = 5.0
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def completion(  # type: ignore[override]
        self,
        prompt: str,
        n_predict: int,
        temperature: float = 0.0,
        cache_prompt: bool = False,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": temperature,
                "cache_prompt": cache_prompt,
            }
        )
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _timings(
    prompt_per_second: float = 250.0, predicted_per_second: float = 35.0
) -> dict[str, Any]:
    return {
        "timings": {
            "prompt_n": 128,
            "prompt_per_second": prompt_per_second,
            "predicted_n": 200,
            "predicted_per_second": predicted_per_second,
        }
    }


def test_pp_tg_probe_reads_timings_directly() -> None:
    # 2 reps × pp probes (different speeds), then 2 reps × tg probes
    client = _FakeClient(
        [
            _timings(prompt_per_second=240.0),
            _timings(prompt_per_second=260.0),
            _timings(predicted_per_second=34.0),
            _timings(predicted_per_second=36.0),
        ]
    )
    result = run_llama_server_benchmark(
        client=client,
        model_id="qwen-7b",
        n_prompt=128,
        n_gen=200,
        repetitions=2,
        probe="pp-tg",
        on_status=lambda s: None,
    )
    assert result.error is None
    approx = pytest.approx  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert result.pp_avg_ts == approx((240.0 + 260.0) / 2)
    assert result.tg_avg_ts == approx((34.0 + 36.0) / 2)
    assert result.backend == "llama-server"
    # pp probes use n_predict=1, tg probes use n_predict=n_gen
    assert client.calls[0]["n_predict"] == 1
    assert client.calls[1]["n_predict"] == 1
    assert client.calls[2]["n_predict"] == 200
    assert client.calls[3]["n_predict"] == 200
    # cache_prompt is disabled so repeated pp probes don't get free prompt processing
    assert all(c["cache_prompt"] is False for c in client.calls)


def test_single_probe_fills_both_pp_and_tg() -> None:
    client = _FakeClient(
        [
            _timings(prompt_per_second=200.0, predicted_per_second=30.0),
            _timings(prompt_per_second=210.0, predicted_per_second=32.0),
        ]
    )
    result = run_llama_server_benchmark(
        client=client,
        model_id="m",
        n_prompt=128,
        n_gen=200,
        repetitions=2,
        probe="single",
        on_status=lambda s: None,
    )
    assert result.error is None
    approx = pytest.approx  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert result.pp_avg_ts == approx((200.0 + 210.0) / 2)
    assert result.tg_avg_ts == approx((30.0 + 32.0) / 2)
    # Single probe always uses n_predict=n_gen
    assert all(c["n_predict"] == 200 for c in client.calls)


def test_probe_records_error_on_http_failure() -> None:
    client = _FakeClient([LlamaServerError("boom"), _timings()])
    result = run_llama_server_benchmark(
        client=client,
        model_id="m",
        n_prompt=128,
        n_gen=200,
        repetitions=1,
        probe="pp-tg",
        on_status=lambda s: None,
    )
    assert result.error is not None
    assert "pp probe failed" in result.error
    # tg probe was not attempted
    assert len(client.calls) == 1


def test_unknown_probe_returns_error() -> None:
    client = _FakeClient([])
    result = run_llama_server_benchmark(
        client=client,
        model_id="m",
        n_prompt=128,
        n_gen=200,
        repetitions=1,
        probe="bogus",
        on_status=lambda s: None,
    )
    assert result.error is not None
    assert "unknown probe" in result.error
    assert client.calls == []


def test_missing_timings_block_records_error() -> None:
    client = _FakeClient([{"content": "ok"}, {"content": "ok"}])  # no timings
    result = run_llama_server_benchmark(
        client=client,
        model_id="m",
        n_prompt=128,
        n_gen=200,
        repetitions=1,
        probe="pp-tg",
        on_status=lambda s: None,
    )
    assert result.error is not None
    assert "no usable timings" in result.error
