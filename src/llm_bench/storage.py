"""Result persistence: ~/.llm-bench/ layout, caching, and run history."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_bench.parser import BenchResult
from llm_bench.sysinfo import SystemInfo

_BASE_DIR = Path.home() / ".llm-bench"
_RESULTS_DIR = _BASE_DIR / "results"


@dataclass
class RunMeta:
    run_id: str
    timestamp: str
    hw_fingerprint: str
    llama_bench_version: str
    n_prompt: int
    n_gen: int
    repetitions: int
    profile_name: str
    model_count: int

    def config_fingerprint(self) -> str:
        raw = (
            f"{self.hw_fingerprint}:{self.n_prompt}:{self.n_gen}"
            f":{self.repetitions}:{self.llama_bench_version}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def model_cache_key(self, hf_repo: str) -> str:
        raw = self.config_fingerprint() + ":" + hf_repo
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunMeta:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _ensure_dirs() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def new_run_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    salt = hashlib.sha256(ts.encode()).hexdigest()[:6]
    return f"{ts}-{salt}"


def make_run_meta(
    run_id: str,
    sysinfo: SystemInfo,
    llama_bench_version: str,
    n_prompt: int,
    n_gen: int,
    repetitions: int,
    profile_name: str,
    model_count: int,
) -> RunMeta:
    return RunMeta(
        run_id=run_id,
        timestamp=datetime.now(UTC).isoformat(),
        hw_fingerprint=sysinfo.fingerprint,
        llama_bench_version=llama_bench_version,
        n_prompt=n_prompt,
        n_gen=n_gen,
        repetitions=repetitions,
        profile_name=profile_name,
        model_count=model_count,
    )


def save_run(meta: RunMeta, results: list[BenchResult], cached_flags: dict[str, bool]) -> Path:
    _ensure_dirs()
    run_dir = _RESULTS_DIR / meta.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2))

    payload = []
    for r in results:
        d = r.to_dict()
        d["_cached"] = cached_flags.get(r.hf_repo, False)
        payload.append(d)
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2))

    return run_dir


def load_run(run_id: str) -> tuple[RunMeta, list[BenchResult], dict[str, bool]]:
    run_dir = _RESULTS_DIR / run_id
    meta = RunMeta.from_dict(json.loads((run_dir / "meta.json").read_text()))
    raw = json.loads((run_dir / "results.json").read_text())
    results = []
    cached: dict[str, bool] = {}
    for d in raw:
        was_cached = d.pop("_cached", False)
        r = BenchResult.from_dict(d)
        results.append(r)
        cached[r.hf_repo] = was_cached
    return meta, results, cached


def list_runs() -> list[RunMeta]:
    _ensure_dirs()
    metas = []
    for run_dir in sorted(_RESULTS_DIR.iterdir(), reverse=True):
        meta_file = run_dir / "meta.json"
        if meta_file.exists():
            try:
                metas.append(RunMeta.from_dict(json.loads(meta_file.read_text())))
            except Exception:
                pass
    return metas


def find_cached_result(meta: RunMeta, hf_repo: str) -> BenchResult | None:
    """
    Search previous runs for a result with the same hardware + config + model.
    Returns the most recent matching BenchResult, or None.
    """
    _ensure_dirs()
    target_key = meta.model_cache_key(hf_repo)
    for run_dir in sorted(_RESULTS_DIR.iterdir(), reverse=True):
        other_meta_file = run_dir / "meta.json"
        results_file = run_dir / "results.json"
        if not other_meta_file.exists() or not results_file.exists():
            continue
        try:
            other_meta = RunMeta.from_dict(json.loads(other_meta_file.read_text()))
            other_key = other_meta.model_cache_key(hf_repo)
            if other_key == target_key:
                for d in json.loads(results_file.read_text()):
                    d.pop("_cached", None)
                    r = BenchResult.from_dict(d)
                    if r.hf_repo == hf_repo and r.error is None:
                        return r
        except Exception:
            continue
    return None
