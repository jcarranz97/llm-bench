# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (runtime + dev)
uv sync

# Run the CLI — three backends supported
uv run llm-bench sysinfo
uv run llm-bench run --backend lm-studio --loaded-only        # LM Studio (recommended)
uv run llm-bench run --backend llama-server                   # llama.cpp llama-server HTTP
uv run llm-bench run --llama-bench /path/to/llama-bench       # llama.cpp llama-bench subprocess
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
uv run pyright .
```

> `PYTHONPATH=""` is required because this machine has ROS installed, which injects broken pytest plugins. The `pyproject.toml` `addopts` disables them by name, but that only takes effect after pytest starts — the empty `PYTHONPATH` prevents the load-time crash.

## Documentation policy

**Whenever the package changes — new flags, new backends, new YAML schema, or any user-visible behaviour change — the documentation must be updated in the same change.** At minimum, check whether each change affects:

- `README.md` — user-facing features, CLI reference, examples, the "How It Works" section, and the GPU-specific profiles section if relevant.
- This file (`AGENTS.md`) — Architecture diagram, Module responsibilities table, Model profiles section, and the per-backend gotchas subsections.

A code change that ships without the corresponding doc update is incomplete. Reviewers should reject PRs that introduce a user-visible change but leave the docs stale.

## Architecture

Three pluggable backends produce the same `BenchResult` shape, so caching, ranking, and `results compare` are backend-agnostic:

- **LM Studio** (the typical path) — drives LM Studio's local HTTP server (`/api/v1/chat`). No subprocess, no compile step. Two probes per model — `pp` (prompt processing speed from TTFT) and `tg` (token generation from `stats.tokens_per_second`) — mirror what `llama-bench` produces.
- **llama-server** — drives llama.cpp's `llama-server` HTTP service via its native `POST /completion` endpoint. Reads `timings.prompt_per_second` and `timings.predicted_per_second` directly — the most precise of the three (no client-side timing, no reasoning-token contamination).
- **llama-bench** — wraps the `llama-bench` binary as a subprocess and parses its JSON output. Used when the user wants to benchmark arbitrary HuggingFace GGUF repos directly without setting up a server.

### Data flow for `llm-bench run`

```
sysinfo.collect()
    → models.select_profile() / get_profile_by_name()       pick the RAM/backend default
    → devices.<backend>_devices()                           probe the active runtime
    → models.matching_gpu_profiles()                        replace default if a GPU profile matches
    → storage.find_cached_result()                          per-model cache lookup
    → backend dispatch (cli.py):
         lm-studio    → lm_studio_runner.run_lm_studio_benchmark()        → BenchResult
         llama-server → llama_server_runner.run_llama_server_benchmark()  → BenchResult
         llama-bench  → runner.run_benchmark() → parser.parse_bench_output() → BenchResult
    → reporter.build_summary_table()
    → storage.save_run()
```

### Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `sysinfo.py` | Reads CPU/RAM via `psutil`; tries `nvidia-smi` / `rocm-smi` / `system_profiler` for GPU. Produces a `SystemInfo` dataclass and a 16-char SHA-256 hardware fingerprint. |
| `models.py` | Loads YAML profiles from `src/llm_bench/data/models/` (bundled) and `~/.llm-bench/models/` (user overrides). Both directories are walked **recursively** so `specific/<vendor>/<card>.yaml` GPU profiles are picked up automatically. `Model` carries optional `hf_repo`, `lm_studio_id`, and `local_path` (path to a local `.gguf`); the loader requires at least one. Optional `extra_args` (list of strings, or a single string we shlex-split) is forwarded verbatim to llama-bench — used for per-model tuning like `-ngl 30`, `-fitt 1024`, `-fa 1`. `Model.identifier` returns whichever is set (priority: lm_studio_id → hf_repo → local_path). `ModelProfile.gpu_match: GpuMatch \| None` flags GPU-specific profiles; `select_profile()` excludes both the `lm_studio` profile *and* any profile with `gpu_match` set, so RAM-based auto-selection only ever returns a generic profile. `matching_gpu_profiles(sysinfo, runtime_devices)` returns the GPU profiles whose `gpu_match` is satisfied by both physical detection and the runtime probe (None means "couldn't probe — proceed with physical-only"). `get_profile_by_name(name)` is used by the LM Studio path to find `lm_studio.yaml` directly. |
| `devices.py` | Probes the active backend for the device list it actually sees. `llama_bench_devices(path)` runs `<llama-bench> --list-devices` and parses the human-readable output (skipping `ggml_*` debug noise and `Device N:` debug lines, only matching the `Available devices:` block). `llama_server_devices(client)` reads `/props.devices` (newer llama.cpp builds only). `lm_studio_runtimes(sysinfo)` runs `lms runtime ls` and synthesises `RuntimeDevice` entries by intersecting installed LM Studio runtimes (e.g. `rocm-llama.cpp`) with detected physical GPUs — LM Studio doesn't expose a per-card device list, this is the closest proxy. Each function returns `None` when the probe fails (older binary, missing endpoint, `lms` not installed) so the matcher can warn-and-proceed. Also exposes `device_index(dev)` (extracts the trailing integer from `ROCm0` / `CUDA1` / etc.) and `auto_env_for_device(dev, vendor)` (vendor → `HIP_VISIBLE_DEVICES` for AMD, `CUDA_VISIBLE_DEVICES` for NVIDIA, `GGML_VK_VISIBLE_DEVICES` for Intel — Apple/Metal returns empty since Metal exposes one device). |
| `lm_studio.py` | Stdlib-only HTTP client (`urllib.request`) for LM Studio's local server. Handles 2026 `/api/v1/models` shape (`{"models": [...]}` with `key` + `loaded_instances`) and falls back to legacy `/api/v0/models` (`{"data": [...]}` with `state`). `chat()` POSTs to `/api/v1/chat` with `{model, system_prompt, input, max_output_tokens}` — `max_tokens` is rejected by the server, so do NOT add it. HTTP errors get the `error.message` field extracted to a single line. |
| `lm_studio_runner.py` | Per-model probe runner. `pp-tg` mode: long prompt with `max_output_tokens=8` (1 fails — TTFT comes back as 0 from the server) → `pp_avg_ts = input_tokens / time_to_first_token_seconds`; short prompt with `max_output_tokens=n_gen` → `tg_avg_ts = stats.tokens_per_second`. `single` mode: one realistic prompt, populates only `tg_avg_ts`. Returns the same `BenchResult` dataclass `parser.py` produces. |
| `llama_server.py` | Stdlib-only HTTP client for llama.cpp's `llama-server`. `props()` returns the server `model_path` / `build_info`; `model_id()` derives a friendly name from `model_alias`, then basename of `model_path`, then `/v1/models`, then a generic fallback. `devices()` returns the `/props.devices` list when the build exposes it (used by `devices.llama_server_devices()` for GPU-profile selection); older builds don't populate it, so the helper returns `None`. `completion()` POSTs to `/completion` with `cache_prompt=False` and `temperature=0.0` so repeated probes don't free-ride on the prefix cache. |
| `llama_server_runner.py` | Per-model probe runner. `pp-tg` mode: long prompt with `n_predict=1` reads `timings.prompt_per_second`; short prompt with `n_predict=n_gen` reads `timings.predicted_per_second`. `single` mode: one realistic prompt populates BOTH metrics from the same response. Same `BenchResult` shape. |
| `runner.py` | Runs `llama-bench` as a subprocess with either `-hf <repo>` (HuggingFace download) or `-m <path>` (local GGUF) — pass exactly one of `hf_repo` / `local_path` to `run_benchmark()`. Uses `Popen` + a background thread to drain stderr and call `on_status(line)` for live progress updates while stdout is captured. `env_vars` overrides (e.g. `HIP_VISIBLE_DEVICES=0`) are layered on top of `os.environ`. |
| `parser.py` | `extract_json()` finds the `[...]` block in stdout even if noise surrounds it. `parse_bench_output()` maps entries where `n_prompt>0,n_gen=0` → pp and `n_gen>0,n_prompt=0` → tg. Defines the `BenchResult` dataclass that both backends produce. |
| `storage.py` | Persists to `~/.llm-bench/results/<run-id>/`. `RunMeta` carries `backend` / `server_url` / `label`. The cache key folds those in **only when non-default**, so existing llama-bench cache keys stay stable across the upgrade. Run dirs are append-only; previous runs are never mutated. |
| `reporter.py` | Pure presentation: rich tables, fit indicators, star ratings. Score = `0.7×TG_normalized + 0.3×PP_normalized`. History table shows `Backend` + `Target` columns so cross-machine LM Studio runs are easy to spot. |
| `cli.py` | Click entry point. `--backend [llama-bench\|lm-studio\|llama-server]` selects the dispatch path. HTTP-backend flags: `--server-url` (backend-specific defaults: 1234 for lm-studio, 8080 for llama-server), `--label`, `--probe`. LM Studio also accepts `--models`, `--all-available`, `--loaded-only`. llama-server accepts `--models` to override the displayed name (one model per server instance, so `--all-available` / `--loaded-only` / `--models-file` are rejected with a friendly error). llama-bench accepts `--models` (CSV of local GGUF paths or HF repo IDs — paths are detected by `.gguf` suffix or existing file) and `--models-file`; `--all-available` / `--loaded-only` are rejected as LM-Studio-only. `--label` is required when `--server-url` is non-localhost (so two machines never collide in the cache). `--env KEY=VALUE` (repeatable) layers env vars onto the llama-bench subprocess — e.g. `--env HIP_VISIBLE_DEVICES=0` to pin AMD GPU 0. Rejected for HTTP backends (the server is already running). Env overrides shard the cache. `--no-gpu-profiles` disables GPU-card-specific profile auto-detection; `--gpu-profile NAME` (repeatable) explicitly selects a GPU profile by name and bypasses auto-detection. The helper `_maybe_swap_in_gpu_profile()` runs after each backend's primary-profile selection and replaces it with a matching GPU profile when one is found. |

### Model profiles (YAML)

Bundled profiles in `src/llm_bench/data/models/`:
- `low_ram.yaml` — < 8 GB RAM (llama.cpp / HuggingFace repos)
- `medium_ram.yaml` — 8–16 GB RAM (llama.cpp / HuggingFace repos)
- `high_ram.yaml` — 16 GB+ RAM (llama.cpp / HuggingFace repos)
- `lm_studio.yaml` — LM Studio model IDs (selected explicitly via `--backend lm-studio`, **not** picked by RAM auto-selection)
- `specific/<vendor>/<card>.yaml` — GPU-card-specific profiles (e.g. `specific/amd/RX7900XT.yaml`), activated when a matching card is detected AND the active runtime sees it

Each YAML has `profile`, `description`, `min_ram_gb`, `max_ram_gb`, and a `models` list. Each model entry must have at least one of `hf_repo` (llama.cpp) or `lm_studio_id` (LM Studio); the loader raises `ValueError` otherwise.

User files placed in `~/.llm-bench/models/` take precedence over bundled ones (matched by `profile` name). Both directories are walked recursively, so user GPU profiles can live at `~/.llm-bench/models/specific/<vendor>/<card>.yaml`. The `--models-file` flag bypasses auto-selection entirely; `--models id1,id2` builds an ad-hoc profile from a CSV string (LM Studio backend only).

#### GPU-card-specific profiles

A YAML under `specific/<vendor>/<card>.yaml` adds a top-level `gpu_match` block:

```yaml
profile: gpu_amd_rx7900xt
description: "AMD Radeon RX 7900 XT (Navi 31, 20 GB VRAM)"
min_ram_gb: 16
max_ram_gb: 9999
gpu_match:
  vendor: amd                                # amd | nvidia | apple | intel
  name_contains: ["RX 7900 XT", "Navi 31"]   # any-of, case-insensitive substring on GpuInfo.name
  backends: [ROCm, Vulkan]                   # any-of; defaults from vendor (amd→ROCm/Vulkan, nvidia→CUDA, apple→Metal, intel→Vulkan)
  min_vram_gb: 18                            # optional VRAM gate
models: [...]
```

When the file lives under `specific/<vendor>/`, `_load_yaml()` enforces that `gpu_match.vendor` matches the parent folder name — guards against typos that would otherwise match nothing.

**Selection is replace-not-append.** When `matching_gpu_profiles()` finds a satisfied profile, the dispatcher in `cli.py` swaps it in for the RAM-based / `lm_studio` default. Multiple matches are sorted by `gpu_match.min_vram_gb` (descending) — the most VRAM-restrictive (most specific) wins. `--no-gpu-profiles` disables the swap; `--gpu-profile NAME` forces a specific one.

**Auto-env (llama-bench backend only).** When a GPU profile is selected, `_maybe_swap_in_gpu_profile()` also returns the specific `RuntimeDevice` it matched (via `models.find_matched_runtime_device`). For the llama-bench backend, the dispatcher then calls `device_probe.auto_env_for_device(matched, vendor)` and layers the result (e.g. `HIP_VISIBLE_DEVICES=0`) into `env_overrides` — but only for keys the user didn't already set with `--env`. The HTTP backends discard the matched device because the env var would not propagate to a remote llama-server / LM Studio process. Auto-set env vars feed into the cache key the same way user-set ones do, so an `HIP_VISIBLE_DEVICES=0` run and an `HIP_VISIBLE_DEVICES=1` run never collide. Static per-card flags (e.g. `-fitt 1024`) belong in the YAML's `extra_args:` field — these are forwarded to the subprocess and key into the cache the same way user-passed extras do.

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

A result is considered valid if `SHA-256(hw_fingerprint + n_prompt + n_gen + repetitions + llama_bench_version + [backend + label + server_url if non-default] + [env_vars if non-empty])` matches a prior run, AND the per-model key `SHA-256(config_fingerprint + hf_repo + [extras if non-empty])` matches. The optional fields are appended only when the run is non-default (i.e. anything other than plain llama-bench with no overrides), so existing run caches from before each addition remain valid. Cross-backend, cross-machine, cross-env, and cross-extra-args runs always get distinct keys. `--fresh` skips all cache lookups.

`extra_args` are stored on each `BenchResult`, so `find_cached_result` can re-key prior results using their own extras — a lookup with `-ngl 30` will not match a cached result that ran with `-ngl 99`.

### Default llama-bench path

`cli.py:DEFAULT_LLAMA_BENCH` is set to `/home/homelab/repos/llama.cpp/build/bin/llama-bench` — the path on the target server, not this dev machine. Only relevant for `--backend llama-bench`.

### LM Studio API gotchas (worth keeping in mind when editing `lm_studio.py`)

- `/api/v1/models` returns `{"models": [...]}` with `key` + `loaded_instances` (not `{"data": [...]}` with `id` + `state` — that's the older `/api/v0` shape).
- A model is "loaded" iff `loaded_instances` is non-empty; the chat-usable identifier is `loaded_instances[0].id` (which usually but not always equals the top-level `key`).
- `/api/v1/chat` rejects unknown keys — sending `max_tokens` triggers HTTP 400 ("unrecognized_keys"). Use `max_output_tokens` only.
- With `max_output_tokens=1`, the server returns `tokens_per_second=0` and `time_to_first_token_seconds=0` (stats are not populated for one-token replies). The pp probe uses `max_output_tokens=8` to work around this.
- Reasoning-capable models emit thinking tokens that count toward `total_output_tokens` and `tokens_per_second`. The system prompt nudges them toward terse output but cannot suppress reasoning entirely; this is a known caveat documented in the runner.

### GPU-specific profile gotchas

- **GPU name strings differ across detection paths.** `lspci -mm` returns something like `AMD Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M]`, while `rocm-smi` returns `Radeon RX 7900 XT`. Use a **list** of substrings in `name_contains` to cover both spellings (`["RX 7900 XT", "Navi 31"]`) instead of a single value.
- **`llama-bench --list-devices` output isn't strict JSON.** It's a human-readable text block whose exact format varies across llama.cpp versions. The parser in `devices.llama_bench_devices()` is regex-based and tolerant of unparseable lines; on parse failure or non-zero exit it returns `None` so the matcher falls back to physical-detection only.
- **LM Studio doesn't expose a per-card device list.** We use `lms runtime ls` (the LM Studio CLI) as a proxy: if `rocm-llama.cpp` is installed, AMD GPUs are presumed usable; if `cuda-llama.cpp` is installed, NVIDIA GPUs are. The synthesised `RuntimeDevice` list is the cross-product of installed runtimes and detected physical GPUs filtered by vendor.
- **`gpu_match.min_vram_gb` treats unknown VRAM as "allow."** On Linux with only `lspci` detection (no `rocm-smi` / `nvidia-smi`), `GpuInfo.vram_gb` can be `None`. The matcher only enforces the threshold when VRAM is known, so partial-detection systems aren't locked out by the gate.
- **A satisfied GPU profile *replaces* the RAM profile.** Don't expect the RAM profile to also run; if the user wants both, they need two `llm-bench run` invocations (one with `--no-gpu-profiles`, one default) or to merge models into the GPU YAML.
- **The matcher excludes `gpu_match` profiles from `select_profile()`.** The exclusion happens in `models.py:select_profile()` — when adding a new automatic-selection criterion, keep the GPU-profile exclusion in place or RAM-based runs will accidentally pull GPU YAMLs and try to run with `-ngl 99` on machines that don't have the card.
- **Device-index enumeration is volatile across machines.** `ROCm0` may be the dGPU on one box and the iGPU on another — it depends on PCIe enumeration and ROCm's discovery order. That's why `auto_env_for_device` reads the index from the matched `RuntimeDevice` rather than hardcoding "0". When refactoring the device parser, preserve the trailing-digit extraction (`re.search(r"(\d+)\s*$", name)`).
- **`llama-bench --list-devices` output is preceded by `ggml_*` debug lines** that look superficially similar (`Device 0: AMD ...`). The parser deliberately rejects those: `Device 0:` doesn't match because of the space between `Device` and the digit, and `ggml_cuda_init: ...` doesn't match because underscores aren't in `[A-Za-z]`. Keep this invariant if rewriting the regex.

### llama-server gotchas (worth keeping in mind when editing `llama_server.py`)

- One model per process — `llama-server` doesn't expose a model list to switch between. `--all-available` / `--loaded-only` / `--models-file` are explicitly rejected for this backend; use `--models <name>` only to override the displayed model name (the actual model loaded is whatever the server was started with).
- `cache_prompt` defaults to `true` server-side — repeated identical prompts will skip prompt processing entirely on the second call, returning `prompt_per_second` of essentially infinity. The client always sends `cache_prompt=false` for probes.
- The connectivity probe in `cli.py` tries `/health` first, then falls back to `/props`. Older `llama-server` builds may not expose `/health`.
- `model_id()` is multi-tier (`model_alias` → basename of `model_path` → `/v1/models` first ID → generic fallback) because different llama.cpp versions populate different fields in `/props`. Don't break the fallback chain when refactoring.
- The `/completion` `timings` block includes both `prompt_per_second` and `predicted_per_second`, so `single` mode reads BOTH from one call (unlike LM Studio's single mode, which can only fill `tg`).
