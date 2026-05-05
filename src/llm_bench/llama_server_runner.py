"""Benchmark runner for llama.cpp's `llama-server`.

Mirrors the pp/tg split that `llama-bench` produces — but reads both metrics
straight off the `timings` block in `/completion` responses, which is more
precise than the LM Studio path:

  pp_avg_ts = timings.prompt_per_second
  tg_avg_ts = timings.predicted_per_second

Probes:
  pp-tg (default):
      * pp probe — long prompt, n_predict=1; only `prompt_per_second` is read.
      * tg probe — short prompt, n_predict=n_gen; only `predicted_per_second` is read.
  single:
      One realistic prompt per repetition with n_predict=n_gen; both metrics
      come from the same call.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any

from llm_bench.llama_server import LlamaServerClient, LlamaServerError
from llm_bench.parser import BenchResult


def _flatten(s: str) -> str:
    return " ".join(s.split())


def _filler_prompt(approx_tokens: int) -> str:
    """Deterministic English filler ~ 4 chars per token."""
    sentence = (
        "Benchmarks measure prompt processing speed by timing how long the model "
        "takes to ingest a fixed number of input tokens before the first output. "
    )
    target_chars = max(approx_tokens, 32) * 4
    parts: list[str] = []
    cur = 0
    while cur < target_chars:
        parts.append(sentence)
        cur += len(sentence)
    return "Repeat the next single word back: " + "".join(parts) + " banana."


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _timings(resp: dict[str, Any]) -> dict[str, Any]:
    t = resp.get("timings")
    return t if isinstance(t, dict) else {}


def run_llama_server_benchmark(
    client: LlamaServerClient,
    model_id: str,
    n_prompt: int,
    n_gen: int,
    repetitions: int,
    probe: str,
    on_status: Callable[[str], None],
) -> BenchResult:
    """Run a llama-server benchmark for the loaded model. Returns a populated BenchResult."""
    result = BenchResult(model_name=model_id, hf_repo=model_id, backend="llama-server")

    if probe not in ("pp-tg", "single"):
        result.error = f"unknown probe '{probe}'"
        return result

    # ── single probe ──────────────────────────────────────────────────────
    if probe == "single":
        pp_speeds: list[float] = []
        tg_speeds: list[float] = []
        for i in range(repetitions):
            on_status(f"single probe {i + 1}/{repetitions}")
            try:
                resp = client.completion(
                    prompt="Tell me a short story about a friendly robot.",
                    n_predict=n_gen,
                )
            except LlamaServerError as e:
                result.error = _flatten(f"single probe failed: {e}")[:120]
                return result
            t = _timings(resp)
            pps = float(t.get("prompt_per_second", 0) or 0)
            tps = float(t.get("predicted_per_second", 0) or 0)
            if pps > 0:
                pp_speeds.append(pps)
            if tps > 0:
                tg_speeds.append(tps)
        pp_avg, pp_std = _mean_std(pp_speeds)
        tg_avg, tg_std = _mean_std(tg_speeds)
        if pp_avg > 0:
            result.pp_avg_ts = pp_avg
            result.pp_std_ts = pp_std
        if tg_avg > 0:
            result.tg_avg_ts = tg_avg
            result.tg_std_ts = tg_std
        if not pp_speeds and not tg_speeds:
            result.error = "single probe returned no timings"
        return result

    # ── pp probe ──────────────────────────────────────────────────────────
    long_prompt = _filler_prompt(n_prompt)
    pp_speeds = []
    for i in range(repetitions):
        on_status(f"pp probe {i + 1}/{repetitions}")
        try:
            resp = client.completion(prompt=long_prompt, n_predict=1)
        except LlamaServerError as e:
            result.error = _flatten(f"pp probe failed: {e}")[:120]
            return result
        t = _timings(resp)
        pps = float(t.get("prompt_per_second", 0) or 0)
        if pps > 0:
            pp_speeds.append(pps)

    # ── tg probe ──────────────────────────────────────────────────────────
    tg_speeds = []
    for i in range(repetitions):
        on_status(f"tg probe {i + 1}/{repetitions}")
        try:
            resp = client.completion(
                prompt="Tell me a short story about a friendly robot.",
                n_predict=n_gen,
            )
        except LlamaServerError as e:
            result.error = _flatten(f"tg probe failed: {e}")[:120]
            return result
        t = _timings(resp)
        tps = float(t.get("predicted_per_second", 0) or 0)
        if tps > 0:
            tg_speeds.append(tps)

    pp_avg, pp_std = _mean_std(pp_speeds)
    tg_avg, tg_std = _mean_std(tg_speeds)
    if pp_avg > 0:
        result.pp_avg_ts = pp_avg
        result.pp_std_ts = pp_std
    if tg_avg > 0:
        result.tg_avg_ts = tg_avg
        result.tg_std_ts = tg_std
    if not pp_speeds and not tg_speeds:
        result.error = "no usable timings in any probe response"
    return result
