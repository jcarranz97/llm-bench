"""Parse llama-bench JSON output into BenchResult objects."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BenchResult:
    model_name: str
    hf_repo: str
    model_type: str = ""
    model_size_bytes: int = 0
    model_n_params: int = 0
    backend: str = ""
    threads: int = 0
    pp_avg_ts: float | None = None
    pp_std_ts: float | None = None
    tg_avg_ts: float | None = None
    tg_std_ts: float | None = None
    error: str | None = None

    @property
    def model_size_gib(self) -> float:
        return self.model_size_bytes / (1024**3)

    @property
    def model_params_b(self) -> float:
        return self.model_n_params / 1e9

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def extract_json(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array from llama-bench stdout, tolerating any surrounding noise."""
    stripped = text.strip()
    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # Find the outermost [...] block
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end > start:
        try:
            result = json.loads(stripped[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return []


def parse_bench_output(
    model_name: str, hf_repo: str, json_data: list[dict[str, Any]]
) -> BenchResult:
    result = BenchResult(model_name=model_name, hf_repo=hf_repo)
    if not json_data:
        result.error = "No data returned"
        return result

    for entry in json_data:
        if not result.model_type:
            result.model_type = entry.get("model_type", "")
            result.model_size_bytes = entry.get("model_size", 0)
            result.model_n_params = entry.get("model_n_params", 0)
            result.threads = entry.get("n_threads", 0)
            backends = entry.get("backends", [])
            if backends:
                result.backend = "/".join(backends)
            elif entry.get("cuda"):
                result.backend = "CUDA"
            elif entry.get("metal"):
                result.backend = "Metal"
            else:
                result.backend = "CPU"

        n_prompt: int = entry.get("n_prompt", 0)
        n_gen: int = entry.get("n_gen", 0)
        avg_ts: float = float(entry.get("avg_ts", 0))
        std_ts: float = float(entry.get("stddev_ts", 0))

        if n_prompt > 0 and n_gen == 0:
            result.pp_avg_ts = avg_ts
            result.pp_std_ts = std_ts
        elif n_gen > 0 and n_prompt == 0:
            result.tg_avg_ts = avg_ts
            result.tg_std_ts = std_ts

    return result
