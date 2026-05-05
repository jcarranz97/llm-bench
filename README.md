# llm-bench

**Real LLM benchmarks on your own hardware — supports [LM Studio](https://lmstudio.ai) and [llama.cpp](https://github.com/ggml-org/llama.cpp).**

`llm-bench` was inspired by [canirun.ai](https://www.canirun.ai): after seeing what models are estimated to run on your system, the natural next step is to measure how they actually perform on your specific machine.

It runs the same kind of test you'd run with `llama-bench` (prompt-processing speed and token-generation speed, repeated and averaged), but you can drive it against either:

- **LM Studio** (recommended) — point it at LM Studio's local HTTP server and benchmark whatever models you've loaded there. Zero compile step. Works across your whole fleet — run it once on each machine and use `llm-bench results compare` to see which one is faster.
- **llama.cpp** — wraps the `llama-bench` binary directly for users who already have a llama.cpp build and want to test arbitrary HuggingFace repos.

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
  - **llama.cpp** built locally — specifically the `llama-bench` binary.

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
   - llama.cpp backend: picks the best-fit YAML from `data/models/` based on RAM tier.
3. **Cache lookup** — each `(hw_fingerprint, params, model)` triple is hashed; matches are shown instantly with a `(cached)` badge. The cache key includes the backend, server URL, and label, so LM Studio results never collide with llama.cpp results, and runs from different machines stay separate.
4. **Benchmark** —
   - LM Studio: POSTs to `/api/v1/chat` per probe, reads the `stats` block.
   - llama.cpp: runs `llama-bench -hf <repo> -p 512 -n 200 -r 5 -o json` and parses stdout.
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

To find the IDs you have loaded in LM Studio:

```bash
curl http://localhost:1234/api/v1/models | jq '.data[].id // .models[].key'
```

## CLI Reference

```
llm-bench run [OPTIONS] [EXTRA_LLAMA_ARGS]...

  Backend selection:
    --backend [llama-bench|lm-studio]   Default: llama-bench

  LM Studio backend:
    --server-url URL          LM Studio base URL (default: http://localhost:1234)
    --label TEXT              Label this machine; required for non-localhost URLs
    --probe [pp-tg|single]    Default: pp-tg
    --models LIST             Comma-separated LM Studio model IDs
    --all-available           Bench every LLM the server knows
    --loaded-only             Bench only currently-loaded models

  llama.cpp backend:
    --llama-bench PATH        Path to llama-bench binary
    -t, --threads N           CPU thread count passed to llama-bench
    --hf-token TOKEN          HuggingFace token (or set HF_TOKEN env var)

  Shared:
    --models-file FILE        Override auto-selected YAML profile
    --fresh                   Ignore cached results
    -r, --repetitions N       Repetitions per probe (default: 5)
    -p, --n-prompt N          Prompt token count for pp probe (default: 512)
    -n, --n-gen N             Generation token count for tg probe (default: 200)
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

## License

MIT
