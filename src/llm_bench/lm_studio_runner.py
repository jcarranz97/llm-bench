"""Benchmark runner for the LM Studio HTTP backend.

Mirrors the pp/tg split that llama-bench produces, so results from this backend
can be compared directly with `llm-bench results compare`.

Probes:
  pp-tg (default):
      * pp probe — long prompt, max_output_tokens=1.
        pp_avg_ts = stats.input_tokens / stats.time_to_first_token_seconds
      * tg probe — short prompt, max_output_tokens=n_gen.
        tg_avg_ts = stats.tokens_per_second
      Each is run `repetitions` times; mean + stddev are stored.

  single:
      One realistic prompt per repetition; only tg_avg_ts is populated.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any

from llm_bench.lm_studio import LMStudioClient, LMStudioError
from llm_bench.parser import BenchResult

# LM Studio reports tokens_per_second=0 / time_to_first_token_seconds=0 when only one
# token is generated. Generate a few so the stats block is populated; pp_avg_ts only
# uses TTFT, so the extra tokens cost a little compute but don't bias the metric.
_PP_MAX_OUTPUT_TOKENS = 8


def _flatten(s: str) -> str:
    return " ".join(s.split())


def _filler_prompt(approx_tokens: int) -> str:
    """A deterministic English-ish prompt sized roughly to `approx_tokens`.

    Uses a rough 4-chars-per-token heuristic. The actual token count comes back
    from the server in `stats.input_tokens` and is what the pp formula uses.
    """
    # Repeat a varied sentence so the tokenizer doesn't dedupe trivially.
    sentence = (
        "Benchmarks measure prompt processing speed by timing how long the model "
        "takes to ingest a fixed number of input tokens before the first output. "
    )
    target_chars = max(approx_tokens, 32) * 4
    out = []
    cur = 0
    while cur < target_chars:
        out.append(sentence)
        cur += len(sentence)
    return "Repeat the next single word back: " + "".join(out) + " banana."


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _stats(resp: dict[str, Any]) -> dict[str, Any]:
    s = resp.get("stats")
    return s if isinstance(s, dict) else {}


def run_lm_studio_benchmark(
    client: LMStudioClient,
    model_id: str,
    n_prompt: int,
    n_gen: int,
    repetitions: int,
    probe: str,
    on_status: Callable[[str], None],
) -> BenchResult:
    """Run an LM Studio benchmark for one model. Returns a populated BenchResult."""
    result = BenchResult(model_name=model_id, hf_repo=model_id, backend="lm-studio")

    if probe not in ("pp-tg", "single"):
        result.error = f"unknown probe '{probe}'"
        return result

    # ── single probe ──────────────────────────────────────────────────────
    if probe == "single":
        tg_speeds: list[float] = []
        for i in range(repetitions):
            on_status(f"single probe {i + 1}/{repetitions}")
            try:
                resp = client.chat(
                    model=model_id,
                    system="Reply concisely.",
                    user_input="What is your favorite color and why?",
                )
            except LMStudioError as e:
                result.error = _flatten(f"single probe failed: {e}")[:120]
                return result
            stats = _stats(resp)
            tps = float(stats.get("tokens_per_second", 0) or 0)
            if tps > 0:
                tg_speeds.append(tps)
        avg, std = _mean_std(tg_speeds)
        if avg > 0:
            result.tg_avg_ts = avg
            result.tg_std_ts = std
        else:
            result.error = "single probe returned no tokens_per_second"
        return result

    # ── pp probe ──────────────────────────────────────────────────────────
    long_prompt = _filler_prompt(n_prompt)
    pp_speeds: list[float] = []
    for i in range(repetitions):
        on_status(f"pp probe {i + 1}/{repetitions}")
        try:
            resp = client.chat(
                model=model_id,
                system="Reply with a single word.",
                user_input=long_prompt,
                max_output_tokens=_PP_MAX_OUTPUT_TOKENS,
            )
        except LMStudioError as e:
            result.error = _flatten(f"pp probe failed: {e}")[:120]
            return result
        stats = _stats(resp)
        input_tokens = float(stats.get("input_tokens", 0) or 0)
        ttft = float(stats.get("time_to_first_token_seconds", 0) or 0)
        if input_tokens > 0 and ttft > 0:
            pp_speeds.append(input_tokens / ttft)

    # ── tg probe ──────────────────────────────────────────────────────────
    tg_speeds = []
    for i in range(repetitions):
        on_status(f"tg probe {i + 1}/{repetitions}")
        try:
            resp = client.chat(
                model=model_id,
                system="Reply with a single word.",
                user_input="Tell me a short story about a friendly robot.",
                max_output_tokens=n_gen,
            )
        except LMStudioError as e:
            result.error = _flatten(f"tg probe failed: {e}")[:120]
            return result
        stats = _stats(resp)
        tps = float(stats.get("tokens_per_second", 0) or 0)
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
        result.error = "no usable stats in any probe response"
    return result
