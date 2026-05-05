# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (runtime + dev)
uv sync

# Run the CLI — both backends supported
uv run llm-bench sysinfo
uv run llm-bench run --backend lm-studio --loaded-only        # LM Studio (recommended)
uv run llm-bench run --llama-bench /path/to/llama-bench       # llama.cpp
uv run llm-bench models list
uv run llm-bench results list

# Run all tests
PYTHONPATH="" uv run pytest tests/ -v

# Run a single test file or test
PYTHONPATH="" uv run pytest tests/test_parser.py -v
PYTHONPATH="" uv run pytest tests/test_lm_studio.py -v
PYTHONPATH="" uv run pytest tests/test_storage.py::test_label_isolates_two_machines -v

# Lint and type-check
uv run ruff check src/ tests/
uv run mypy src/
```

> `PYTHONPATH=""` is required because this machine has ROS installed, which injects broken pytest plugins. The `pyproject.toml` `addopts` disables them by name, but that only takes effect after pytest starts — the empty `PYTHONPATH` prevents the load-time crash.

## Architecture

Two pluggable backends produce the same `BenchResult` shape, so caching, ranking, and `results compare` are backend-agnostic:

- **LM Studio** (the typical path) — drives LM Studio's local HTTP server (`/api/v1/chat`). No subprocess, no compile step. Two probes per model — `pp` (prompt processing speed from TTFT) and `tg` (token generation from `stats.tokens_per_second`) — mirror what `llama-bench` produces.
- **llama.cpp** — wraps the `llama-bench` binary as a subprocess and parses its JSON output. Used when the user wants to benchmark arbitrary HuggingFace GGUF repos directly.

### Data flow for `llm-bench run`

```
sysinfo.collect()
    → models.select_profile() / get_profile_by_name()    pick the model list
    → storage.find_cached_result()                       per-model cache lookup
    → backend dispatch (cli.py):
         lm-studio  → lm_studio_runner.run_lm_studio_benchmark()  → BenchResult
         llama.cpp  → runner.run_benchmark() → parser.parse_bench_output()  → BenchResult
    → reporter.build_summary_table()
    → storage.save_run()
```

### Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `sysinfo.py` | Reads CPU/RAM via `psutil`; tries `nvidia-smi` / `rocm-smi` / `system_profiler` for GPU. Produces a `SystemInfo` dataclass and a 16-char SHA-256 hardware fingerprint. |
| `models.py` | Loads YAML profiles from `src/llm_bench/data/models/` (bundled) and `~/.llm-bench/models/` (user overrides). `Model` carries optional `hf_repo` *and* `lm_studio_id`; `Model.identifier` returns whichever is set. `select_profile()` picks the highest-tier RAM profile (excluding the `lm_studio` profile). `get_profile_by_name(name)` is used by the LM Studio path to find `lm_studio.yaml` directly. |
| `lm_studio.py` | Stdlib-only HTTP client (`urllib.request`) for LM Studio's local server. Handles 2026 `/api/v1/models` shape (`{"models": [...]}` with `key` + `loaded_instances`) and falls back to legacy `/api/v0/models` (`{"data": [...]}` with `state`). `chat()` POSTs to `/api/v1/chat` with `{model, system_prompt, input, max_output_tokens}` — `max_tokens` is rejected by the server, so do NOT add it. HTTP errors get the `error.message` field extracted to a single line. |
| `lm_studio_runner.py` | Per-model probe runner. `pp-tg` mode: long prompt with `max_output_tokens=8` (1 fails — TTFT comes back as 0 from the server) → `pp_avg_ts = input_tokens / time_to_first_token_seconds`; short prompt with `max_output_tokens=n_gen` → `tg_avg_ts = stats.tokens_per_second`. `single` mode: one realistic prompt, populates only `tg_avg_ts`. Returns the same `BenchResult` dataclass `parser.py` produces. |
| `runner.py` | Runs `llama-bench -hf <repo> -o json ...`. Uses `Popen` + a background thread to drain stderr and call `on_status(line)` for live progress updates while stdout is captured. |
| `parser.py` | `extract_json()` finds the `[...]` block in stdout even if noise surrounds it. `parse_bench_output()` maps entries where `n_prompt>0,n_gen=0` → pp and `n_gen>0,n_prompt=0` → tg. Defines the `BenchResult` dataclass that both backends produce. |
| `storage.py` | Persists to `~/.llm-bench/results/<run-id>/`. `RunMeta` carries `backend` / `server_url` / `label`. The cache key folds those in **only when non-default**, so existing llama-bench cache keys stay stable across the upgrade. Run dirs are append-only; previous runs are never mutated. |
| `reporter.py` | Pure presentation: rich tables, fit indicators, star ratings. Score = `0.7×TG_normalized + 0.3×PP_normalized`. History table shows `Backend` + `Target` columns so cross-machine LM Studio runs are easy to spot. |
| `cli.py` | Click entry point. `--backend [llama-bench\|lm-studio]` selects the dispatch path. LM Studio flags: `--server-url`, `--label`, `--probe`, `--models`, `--all-available`, `--loaded-only`. `--label` is required when `--server-url` is non-localhost (so two machines never collide in the cache). |

### Model profiles (YAML)

Bundled profiles in `src/llm_bench/data/models/`:
- `low_ram.yaml` — < 8 GB RAM (llama.cpp / HuggingFace repos)
- `medium_ram.yaml` — 8–16 GB RAM (llama.cpp / HuggingFace repos)
- `high_ram.yaml` — 16 GB+ RAM (llama.cpp / HuggingFace repos)
- `lm_studio.yaml` — LM Studio model IDs (selected explicitly via `--backend lm-studio`, **not** picked by RAM auto-selection)

Each YAML has `profile`, `description`, `min_ram_gb`, `max_ram_gb`, and a `models` list. Each model entry must have at least one of `hf_repo` (llama.cpp) or `lm_studio_id` (LM Studio); the loader raises `ValueError` otherwise.

User files placed in `~/.llm-bench/models/` take precedence over bundled ones (matched by `profile` name). The `--models-file` flag bypasses auto-selection entirely; `--models id1,id2` builds an ad-hoc profile from a CSV string (LM Studio backend only).

### Result storage layout

```
~/.llm-bench/
├── models/                    ← user YAML overrides
└── results/
    └── YYYYMMDD-HHMMSS-<hash>/
        ├── meta.json          ← RunMeta (hw fingerprint, backend, server_url, label, params, timestamp)
        └── results.json       ← list of BenchResult dicts + _cached flag
```

### Cache invalidation

A result is considered valid if `SHA-256(hw_fingerprint + n_prompt + n_gen + repetitions + llama_bench_version + [backend + label + server_url if non-default])` matches a prior run. The new fields are appended only when the run is non-default (i.e. anything other than plain llama-bench), so existing run caches from before the LM Studio backend was added remain valid. Cross-backend and cross-machine runs always get distinct keys. `--fresh` skips all cache lookups.

### Default llama-bench path

`cli.py:DEFAULT_LLAMA_BENCH` is set to `/home/homelab/repos/llama.cpp/build/bin/llama-bench` — the path on the target server, not this dev machine. Only relevant for `--backend llama-bench`.

### LM Studio API gotchas (worth keeping in mind when editing `lm_studio.py`)

- `/api/v1/models` returns `{"models": [...]}` with `key` + `loaded_instances` (not `{"data": [...]}` with `id` + `state` — that's the older `/api/v0` shape).
- A model is "loaded" iff `loaded_instances` is non-empty; the chat-usable identifier is `loaded_instances[0].id` (which usually but not always equals the top-level `key`).
- `/api/v1/chat` rejects unknown keys — sending `max_tokens` triggers HTTP 400 ("unrecognized_keys"). Use `max_output_tokens` only.
- With `max_output_tokens=1`, the server returns `tokens_per_second=0` and `time_to_first_token_seconds=0` (stats are not populated for one-token replies). The pp probe uses `max_output_tokens=8` to work around this.
- Reasoning-capable models emit thinking tokens that count toward `total_output_tokens` and `tokens_per_second`. The system prompt nudges them toward terse output but cannot suppress reasoning entirely; this is a known caveat documented in the runner.
