# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (runtime + dev)
uv sync

# Run the CLI
uv run llm-bench sysinfo
uv run llm-bench run --llama-bench /path/to/llama-bench
uv run llm-bench models list
uv run llm-bench results list

# Run all tests
PYTHONPATH="" uv run pytest tests/ -v

# Run a single test file or test
PYTHONPATH="" uv run pytest tests/test_parser.py -v
PYTHONPATH="" uv run pytest tests/test_storage.py::test_find_cached_result -v

# Lint and type-check
uv run ruff check src/ tests/
uv run mypy src/
```

> `PYTHONPATH=""` is required because this machine has ROS installed, which injects broken pytest plugins. The `pyproject.toml` `addopts` disables them by name, but that only takes effect after pytest starts — the empty `PYTHONPATH` prevents the load-time crash.

## Architecture

The tool wraps `llama-bench` (from llama.cpp) to run real LLM benchmarks, auto-selects a model list based on detected RAM, and caches results so re-runs skip already-benchmarked models.

### Data flow for `llm-bench run`

```
sysinfo.collect()
    → models.select_profile()       picks the right YAML file
    → storage.find_cached_result()  per-model cache lookup
    → runner.run_benchmark()        subprocess + live stderr streaming
    → parser.parse_bench_output()   JSON → BenchResult
    → reporter.build_summary_table()
    → storage.save_run()
```

### Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `sysinfo.py` | Reads CPU/RAM via `psutil`; tries `nvidia-smi` / `rocm-smi` / `system_profiler` for GPU. Produces a `SystemInfo` dataclass and a 16-char SHA-256 hardware fingerprint. |
| `models.py` | Loads YAML profiles from `src/llm_bench/data/models/` (bundled) and `~/.llm-bench/models/` (user overrides). `select_profile()` picks the highest-tier profile whose `min_ram_gb ≤ total_ram`. |
| `runner.py` | Runs `llama-bench -hf <repo> -o json ...`. Uses `Popen` + a background thread to drain stderr and call `on_status(line)` for live progress updates while stdout is captured. |
| `parser.py` | `extract_json()` finds the `[...]` block in stdout even if noise surrounds it. `parse_bench_output()` maps entries where `n_prompt>0,n_gen=0` → pp and `n_gen>0,n_prompt=0` → tg. |
| `storage.py` | Persists to `~/.llm-bench/results/<run-id>/`. The per-model cache key is `SHA-256(hw_fingerprint + params + llama_bench_version + hf_repo)`. Run dirs are append-only; previous runs are never mutated. |
| `reporter.py` | Pure presentation: rich tables, fit indicators, star ratings. Score = `0.7×TG_normalized + 0.3×PP_normalized`. |
| `cli.py` | Click entry point. `llm-bench run` is the main command; `sysinfo`, `models list`, `results list/show/compare` are subcommands. |

### Model profiles (YAML)

Bundled profiles in `src/llm_bench/data/models/`:
- `low_ram.yaml` — < 8 GB RAM
- `medium_ram.yaml` — 8–16 GB RAM  
- `high_ram.yaml` — 16 GB+ RAM

Each YAML has `profile`, `description`, `min_ram_gb`, `max_ram_gb`, and a `models` list. User files placed in `~/.llm-bench/models/` take precedence over bundled ones (matched by `profile` name). The `--models-file` flag bypasses auto-selection entirely.

### Result storage layout

```
~/.llm-bench/
├── models/                    ← user YAML overrides
└── results/
    └── YYYYMMDD-HHMMSS-<hash>/
        ├── meta.json          ← RunMeta (hw fingerprint, params, timestamp)
        └── results.json       ← list of BenchResult dicts + _cached flag
```

### Cache invalidation

A result is considered valid if `SHA-256(hw_fingerprint + n_prompt + n_gen + repetitions + llama_bench_version + hf_repo)` matches a prior run. Changing any of these — including upgrading llama-bench — triggers a re-benchmark. `--fresh` skips all cache lookups.

### Default llama-bench path

`cli.py:DEFAULT_LLAMA_BENCH` is set to `/home/homelab/repos/llama.cpp/build/bin/llama-bench` — the path on the target server, not this dev machine.
