# llm-bench

**Real LLM benchmarks on your own hardware — supports [LM Studio](https://lmstudio.ai) and [llama.cpp](https://github.com/ggml-org/llama.cpp).**

`llm-bench` was inspired by [canirun.ai](https://www.canirun.ai): after seeing what models are estimated to run on your system, the natural next step is to measure how they actually perform on your specific machine.

It runs the same kind of test you'd run with `llama-bench` (prompt-processing speed and token-generation speed, repeated and averaged), but you can drive it against any of:

- **LM Studio** (recommended for most people) — point it at LM Studio's local HTTP server and benchmark whatever models you've loaded there. Zero compile step. Works across your whole fleet — run it once on each machine and use `llm-bench results compare` to see which one is faster.
- **llama.cpp `llama-server`** — point it at a running `llama-server` HTTP service. Native `/completion` endpoint exposes precise per-request timings, no client-side timing or token-counting needed.
- **llama.cpp `llama-bench`** — wraps the `llama-bench` binary directly for users who already have a llama.cpp build and want to test arbitrary HuggingFace repos.

```
System Info
┌──────────────────────────────────────────────────────────┐
│ CPU   AMD Ryzen 9 7950X  (16 cores, 32 threads)          │
│ RAM   ████████░░░░░░░░░░  16.6 GB free / 30.0 GB total   │
│ GPU   No GPU detected — running on CPU                   │
│ OS    Linux 7.0.0                                        │
│                                                          │
│ Selected profile:  LM Studio HTTP backend — 2026-era models │
└──────────────────────────────────────────────────────────┘

 LLM Benchmark Results
╭───┬───────────────────────────┬────────┬──────────┬──────────┬───────┬───────╮
│ # │ Model                     │ Fits?  │ Backend  │ PP t/s   │ TG t/s│ Score │
├───┼───────────────────────────┼────────┼──────────┼──────────┼───────┼───────┤
│ 1 │ qwen/qwen3.6-35b-a3b      │ ✓ Fits │ lm-studio│ 1655.6   │ 31.88 │ 100.0 │
╰───┴───────────────────────────┴────────┴──────────┴──────────┴───────┴───────╯
```

## Requirements

- Python 3.11+
- One of:
  - **LM Studio** with the local server enabled (Settings → Developer → "Serve on local network"), and at least one model loaded.
  - **llama.cpp `llama-server`** running on its default port (or any URL you pass via `--server-url`). Start it with `./llama-server -m model.gguf --port 8080` (and add `-ngl 999` for full GPU offload, etc.).
  - **llama.cpp `llama-bench`** binary built locally.

## Installation

### With uv (recommended)

```bash
uv tool install llm-bench
```

### With pip

```bash
pip install llm-bench
```

### From source (development)

```bash
git clone https://github.com/jcarranz/llm-bench
cd llm-bench
uv sync
uv run llm-bench --help
```

## Quick Start — LM Studio (recommended)

```bash
# Bench every model currently loaded in LM Studio
llm-bench run --backend lm-studio --loaded-only

# Bench every LLM the server knows about (LM Studio JIT-loads them on first request)
llm-bench run --backend lm-studio --all-available

# Bench a specific list
llm-bench run --backend lm-studio --models qwen/qwen3.6-35b-a3b,google/gemma-4-e4b
```

### Compare two machines

This is the canonical use case — you have LM Studio running on a desktop and a homelab box, both with the same models loaded, and want to know which one is faster:

```bash
# On each machine, pointing the client at the local server
desktop$  llm-bench run --backend lm-studio --loaded-only
homelab$  llm-bench run --backend lm-studio --loaded-only

# Or run from one machine, hitting both over LAN (note --label, required for non-localhost):
$  llm-bench run --backend lm-studio --server-url http://desktop:1234 --label desktop --loaded-only
$  llm-bench run --backend lm-studio --server-url http://homelab:1234 --label homelab --loaded-only

# Then sync the results dirs and compare
$  llm-bench results compare <desktop-run-id> <homelab-run-id>
```

`--label` makes the cache key for each box distinct, so results from different machines never collide even if they appear to share hardware fingerprints.

### Probes

By default `--probe pp-tg` runs two probes per model and reports both, mirroring `llama-bench`:

- **pp** (prompt processing) — sends a long prompt with `max_output_tokens=8` and computes `input_tokens / time_to_first_token_seconds`. Measures how fast the model ingests context.
- **tg** (token generation) — sends a short prompt and reads `stats.tokens_per_second` from the response. Measures how fast it streams tokens.

Use `--probe single` for a quick one-shot ("what's your favorite color?") that reports only TG — useful for a sanity check.

## Quick Start — llama-server (llama.cpp HTTP server)

If you've started `llama-server` from llama.cpp (e.g. `./llama-server -m model.gguf --port 8080`), point `llm-bench` at it the same way you would LM Studio. The model is auto-discovered from `/props`, and metrics come straight from the `timings` block of `/completion` — no client-side timing.

```bash
# Default: hits localhost:8080, auto-detects the loaded model
llm-bench run --backend llama-server

# Different port / host
llm-bench run --backend llama-server --server-url http://localhost:9090

# Quick one-shot probe
llm-bench run --backend llama-server --probe single --repetitions 3

# Override the displayed model name (defaults to /props's model_alias or basename of model_path)
llm-bench run --backend llama-server --models my-7b-q4
```

### Compare two boxes running llama-server

```bash
# On each machine, default localhost:8080
desktop$  llm-bench run --backend llama-server
homelab$  llm-bench run --backend llama-server

# Or remotely from one machine, hitting both over LAN (--label required for non-localhost):
$  llm-bench run --backend llama-server --server-url http://desktop:8080 --label desktop
$  llm-bench run --backend llama-server --server-url http://homelab:8080 --label homelab

$  llm-bench results compare <desktop-run-id> <homelab-run-id>
```

`llama-server` runs one model per process, so flags like `--all-available`, `--loaded-only`, and `--models-file` aren't accepted for this backend. To benchmark several models, run multiple `llama-server` instances on different ports and run `llm-bench` against each.

### Why prefer `llama-server` over `lm-studio` when available

The `/completion` response includes a `timings` block with `prompt_per_second` and `predicted_per_second` already computed. That's a direct, precise read of pp/tg speed — no need to deal with reasoning-token contamination of `tokens_per_second` or with LM Studio's quirk of returning `time_to_first_token=0` for very short generations. If you're already running `llama-server`, this is the most accurate backend.

## Quick Start — llama.cpp

```bash
# Auto-detects RAM, picks a profile, downloads + benchmarks each model from HuggingFace
llm-bench run --llama-bench /path/to/llama.cpp/build/bin/llama-bench

# Browse past runs
llm-bench results list
llm-bench results show 20240501-143022-abc123

# Compare two runs (e.g. before/after adding RAM, or two different llama.cpp builds)
llm-bench results compare 20240501-143022-abc123 20240601-092011-def456
```

The llama.cpp backend uses bundled YAML profiles (`low_ram.yaml`, `medium_ram.yaml`, `high_ram.yaml`) and downloads models from HuggingFace on first use.

## How It Works

1. **Hardware detection** — reads CPU model/cores, total/available RAM, and GPU (NVIDIA via `nvidia-smi`, AMD via `rocm-smi`, Apple via `system_profiler`).
2. **Profile / model selection**:
   - LM Studio backend: defaults to the bundled `lm_studio.yaml`, but `--loaded-only` and `--all-available` discover models from the running server.
   - llama-server backend: auto-discovers the single loaded model from `/props`. Override the displayed name with `--models <id>`.
   - llama-bench backend: picks the best-fit YAML from `data/models/` based on RAM tier.
   - **GPU-card-specific profiles** (e.g. AMD RX 7900 XT) override the RAM/backend default when the card is both physically detected *and* the active runtime reports it as available — see [GPU-specific profiles](#gpu-specific-profiles). Pass `--no-gpu-profiles` to skip this, or `--gpu-profile NAME` to pick one explicitly.
3. **Cache lookup** — each `(hw_fingerprint, params, model)` triple is hashed; matches are shown instantly with a `(cached)` badge. The cache key includes the backend, server URL, and label, so results from different backends or different machines never collide.
4. **Benchmark** —
   - LM Studio: POSTs to `/api/v1/chat` per probe, reads the `stats` block.
   - llama-server: POSTs to `/completion` per probe, reads `timings.prompt_per_second` and `timings.predicted_per_second` directly.
   - llama-bench: runs `llama-bench -hf <repo> -p 512 -n 200 -r 5 -o json` and parses stdout.
5. **Rating** — `score = 0.7 × (TG t/s ÷ best) + 0.3 × (PP t/s ÷ best)`, scaled 0–100 and shown as 1–5 stars. TG is weighted more because it determines chat response speed.
6. **Save** — writes results to `~/.llm-bench/results/<run-id>/` for future comparisons.

## Custom Model Profiles

Place YAML files in `~/.llm-bench/models/` to override or extend the bundled profiles:

```yaml
profile: my_lm_studio_models
description: "What I have loaded in LM Studio"
min_ram_gb: 0
max_ram_gb: 9999

models:
  # LM Studio entries use lm_studio_id (the value from `/api/v1/models`):
  - name: "Qwen 3.6 35B-A3B"
    lm_studio_id: "qwen/qwen3.6-35b-a3b"
    estimated_size_gb: 22.0
    tags: [moe, qwen]

  # llama.cpp entries use hf_repo (HuggingFace GGUF repo):
  - name: "Llama 3.3 70B"
    hf_repo: "ggml-org/Llama-3.3-70B-Instruct-GGUF"
    estimated_size_gb: 42.5
    tags: [large, llama]
```

Each entry must specify at least one of `lm_studio_id` or `hf_repo`. Pass a file directly with `--models-file ~/my-models.yaml`, or rely on profile auto-discovery (matched by `profile` name).

For per-GPU-card profiles (e.g. a model list tuned for an AMD RX 7900 XT or NVIDIA RTX 4090), drop the YAML into `~/.llm-bench/models/specific/<vendor>/<card>.yaml` and add a `gpu_match` block — see [GPU-specific profiles](#gpu-specific-profiles).

To find the IDs you have loaded in LM Studio:

```bash
curl http://localhost:1234/api/v1/models | jq '.data[].id // .models[].key'
```

## GPU-specific profiles

Some workloads only make sense on a particular card — e.g. a 35B model that fits on an RX 7900 XT's 20 GB VRAM but won't on a 12 GB card. Drop a YAML under `specific/<vendor>/<card>.yaml` to define a profile that activates only when that exact card is present:

```
src/llm_bench/data/models/             ← bundled
    specific/
        amd/RX7900XT.yaml
        nvidia/RTX4090.yaml          ← (future)
~/.llm-bench/models/                   ← user overrides (same layout)
    specific/
        amd/RX7900XT.yaml
```

Each file uses the regular profile schema plus a `gpu_match:` block:

```yaml
profile: gpu_amd_rx7900xt
description: "AMD Radeon RX 7900 XT (Navi 31, 20 GB VRAM) — full-offload models"
min_ram_gb: 16
max_ram_gb: 9999

gpu_match:
  vendor: amd                                # amd | nvidia | apple | intel
  name_contains: ["RX 7900 XT", "Navi 31"]   # any-of, case-insensitive substring
  backends: [ROCm, Vulkan]                   # restrict to a runtime; optional
  min_vram_gb: 18                            # optional VRAM threshold

models:
  - name: "Qwen3.6-35B-A3B"
    hf_repo: "lmstudio-community/Qwen3.6-35B-A3B-GGUF"
    estimated_size_gb: 9.3
    extra_args: "-ngl 99"                    # full GPU offload — safe here
```

**Selection rules.** A GPU-specific profile **replaces** the RAM-based / backend default when *both* of the following are true:

1. **Physical detection.** Some `GpuInfo` entry from `nvidia-smi` / `rocm-smi` / `system_profiler` / `lspci` matches the `gpu_match` block (vendor + at least one `name_contains` substring + VRAM threshold).
2. **Runtime confirmation.** The active backend itself sees the card as usable:
   - `--backend llama-bench` → `llama-bench --list-devices` lists a matching device.
   - `--backend llama-server` → `/props.devices` (newer builds) lists a matching device.
   - `--backend lm-studio` → `lms runtime ls` shows a runtime whose backend matches the card's vendor (e.g. `rocm-llama.cpp` for AMD).

When the runtime probe can't run (older binary that doesn't accept `--list-devices`, missing endpoint, `lms` not installed), `llm-bench` warns and proceeds with physical detection only — the goal is to never silently disable GPU profiles because of a missing query.

**Auto-pinned device env var (llama-bench backend).** When a GPU profile activates and the runtime probe identifies which device it bound to (e.g. `ROCm0` on a system where the dGPU is index 0, but `ROCm1` where it's index 1), `llm-bench` auto-sets the corresponding pinning env var for the llama-bench subprocess:

| Vendor | Auto-set env var |
|--------|------------------|
| `amd` | `HIP_VISIBLE_DEVICES=<index>` |
| `nvidia` | `CUDA_VISIBLE_DEVICES=<index>` |
| `intel` | `GGML_VK_VISIBLE_DEVICES=<index>` |
| `apple` | (none — Metal exposes a single device) |

User-passed `--env KEY=VALUE` always wins over the auto-detect, so you can override on a per-run basis. The auto-set env var also shards the result cache (different indexes are different runs).

For tuning flags that are static per card (e.g. `-fitt 1024` to leave a 1 GB VRAM margin), put them in the YAML's `extra_args:` field. The bundled `gpu_amd_rx7900xt` profile uses `extra_args: ["-fitt", "1024"]` so the largest models fit on 20 GB without OOM.

**Flags.**

```bash
# Default: auto-pick the matching GPU profile if any
llm-bench run

# Skip the auto-detection
llm-bench run --no-gpu-profiles

# Pick a specific profile by name
llm-bench run --gpu-profile gpu_amd_rx7900xt
```

You can preview which GPU profile would activate on the current machine with `llm-bench sysinfo` (it lists every GPU profile whose physical-detection check passes — actual selection at `run` time also requires the backend runtime to see the card).

## CLI Reference

```
llm-bench run [OPTIONS] [EXTRA_LLAMA_ARGS]...

  Backend selection:
    --backend [llama-bench|lm-studio|llama-server]   Default: llama-bench

  HTTP backends (lm-studio + llama-server):
    --server-url URL          Base URL. Default: lm-studio → http://localhost:1234,
                                                  llama-server → http://localhost:8080
    --label TEXT              Label this machine; required for non-localhost URLs
    --probe [pp-tg|single]    Default: pp-tg

  LM Studio only:
    --models LIST             Comma-separated LM Studio model IDs
    --all-available           Bench every LLM the server knows
    --loaded-only             Bench only currently-loaded models

  llama-server only:
    --models NAME             Override the displayed model name (one model per server instance)

  llama-bench backend only:
    --llama-bench PATH        Path to llama-bench binary
    -t, --threads N           CPU thread count passed to llama-bench
    --hf-token TOKEN          HuggingFace token (or set HF_TOKEN env var)

  Shared:
    --models-file FILE        Override auto-selected YAML profile (lm-studio + llama-bench only)
    --fresh                   Ignore cached results
    -r, --repetitions N       Repetitions per probe (default: 5)
    -p, --n-prompt N          Prompt token count for pp probe (default: 512)
    -n, --n-gen N             Generation token count for tg probe (default: 200)
    --no-gpu-profiles         Disable auto-pick of GPU-card-specific profiles
    --gpu-profile NAME        Run the named GPU-specific profile (repeatable)
    -o, --output FORMAT       table | json | markdown

  Any extra options after `--` are forwarded verbatim to llama-bench:
    llm-bench run -- -fa 1 -ngl 32

llm-bench sysinfo                       Show hardware info and selected profile
llm-bench models list                   List all available model profiles
llm-bench results list                  List past runs (shows backend + target column)
llm-bench results show RUN_ID
llm-bench results compare RUN_ID_A RUN_ID_B
```

## Result Storage

```
~/.llm-bench/
├── models/                   ← your custom YAML files (optional)
└── results/
    └── 20240501-143022-abc123/
        ├── meta.json         ← hw fingerprint, backend, server_url, label, params
        └── results.json      ← benchmark results per model
```

Results are **never mutated** — each run creates a new directory. The cache lookup is keyed on `SHA256(hw_fingerprint + params + backend + label + server_url + model_id)`, so changing any parameter (or the backend, or the target machine) triggers a fresh benchmark.

## Development

### Setup

```bash
git clone https://github.com/jcarranz/llm-bench
cd llm-bench
uv sync                          # creates .venv and installs all deps
uv run llm-bench --help          # verify the install
```

### Running tests

```bash
PYTHONPATH="" uv run pytest tests/ -v                               # all tests
PYTHONPATH="" uv run pytest tests/test_lm_studio.py -v              # one file
PYTHONPATH="" uv run pytest tests/test_storage.py::test_label_isolates_two_machines -v
```

### Lint and type-check

```bash
uv run ruff check src/ tests/
uv run mypy src/
```

### Building a distributable package

```bash
uv build
```

This produces two files in `dist/`:

```
dist/
  llm_bench-0.1.0-py3-none-any.whl   ← preferred for installing
  llm_bench-0.1.0.tar.gz             ← source distribution
```

Copy either file to another machine and install it:

```bash
# With uv (installs as a standalone tool, adds llm-bench to PATH)
uv tool install llm_bench-0.1.0-py3-none-any.whl

# With pip
pip install llm_bench-0.1.0-py3-none-any.whl
```

## FAQ

### Why does a model that loads fine in LM Studio / `llama-cli` fail with `failed to create context with model` in llm-bench?

You'll see something like this in the results table:

```
[1/1] ⚠ Qwen3.6-35B-A3B  exit 1: main: error: failed to create context with model '…'
```

This is **llm-bench-specific** — or, more precisely, specific to the `llama-bench` backend. It happens when the model is too big to fully offload onto your GPU(s).

**Why it's specific to llama-bench:** the underlying `llama-bench` binary defaults to `-ngl 99`, meaning *put every layer on the GPU*. `llama-cli` and LM Studio do not — `llama-cli` defaults to `-ngl 0` (CPU only) unless you pass `-ngl`, and LM Studio auto-picks how many layers fit. So a 21 GB Q4 model loads fine in LM Studio on a 20 GB GPU (it offloads a partial set of layers and keeps the rest on CPU), but the same model in llama-bench tries to put all 21 GB on a 20 GB device and fails during context allocation.

**Fix:** pass `-fitt <margin_MiB>` through to llama-bench so it auto-fits the model to available VRAM with the given margin (in MiB) per device:

```bash
# Auto-fit the model to fit your GPU with 1 GB headroom
uv run llm-bench run \
  --env HIP_VISIBLE_DEVICES=0 \
  --llama-bench /path/to/llama.cpp/build/bin/llama-bench \
  -- -fitt 1024
```

Anything after the bare `--` is forwarded verbatim to `llama-bench`. Bump the margin if you still OOM on the smaller end, or lower it (e.g. `-fitt 256`) if you want to maximize VRAM usage. See `llama-bench --help` for related flags (`-fitc`, `-ngl`, `-ncmoe`).

**Make it permanent for a specific model** by adding `extra_args` to your YAML profile (see [Custom Model Profiles](#custom-model-profiles)):

```yaml
models:
  - name: "Qwen3.6-35B-A3B"
    hf_repo: "lmstudio-community/Qwen3.6-35B-A3B-GGUF"
    estimated_size_gb: 21.0
    extra_args: ["-fitt", "1024"]   # auto-fit; per-model
```

The `extra_args` field shards the result cache, so cached results from a run with different `extra_args` won't bleed into your tuned run.

### Why does the same error not happen with `--backend lm-studio` or `--backend llama-server`?

Those backends drive an already-loaded model on a server you started yourself. Whatever offload / context decisions LM Studio or `llama-server` made at startup are baked in by the time llm-bench connects. The `llama-bench` backend, by contrast, spawns a fresh subprocess per model, and that's where llama-bench's `-ngl 99` default bites you.

## License

MIT
