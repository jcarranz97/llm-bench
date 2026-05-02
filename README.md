# llm-bench

**Real LLM benchmarks on your own hardware — inspired by [canirun.ai](https://www.canirun.ai).**

`llm-bench` was inspired by [canirun.ai](https://www.canirun.ai): after seeing what models are estimated to run on your system, the natural next step is to measure how they actually perform on your specific machine. This tool does exactly that — it uses [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-bench` to run real benchmarks, auto-detects your hardware, picks a suitable set of models, and caches results so re-runs are instant.

Think of it as a hands-on complement: check what `canirun.ai` says fits your hardware, then use `llm-bench` to get real numbers on your machine with any model you want to try.

```
System Info
┌──────────────────────────────────────────────────────────┐
│ CPU   Intel Core i7-12700  (20 threads)                  │
│ RAM   ████████░░░░░░░░░░░░  9.3 GB free / 16.0 GB total  │
│ GPU   No GPU detected — running on CPU                   │
│ OS    Linux 5.15.0                                       │
│                                                          │
│ Selected profile:  Systems with 8–16 GB RAM              │
└──────────────────────────────────────────────────────────┘

 LLM Benchmark Results
╭───┬──────────────────────────────┬────────┬──────────┬─────────┬──────────┬────────┬──────────┬──────────┬───────┬──────────────╮
│ # │ Model                        │ Fits?  │ Size     │ Params  │ Backend  │Threads │ PP t/s   │ TG t/s   │ Score │ Rating       │
├───┼──────────────────────────────┼────────┼──────────┼─────────┼──────────┼────────┼──────────┼──────────┼───────┼──────────────┤
│ 1 │ Gemma 4 2B RotorQuant Q4_K_M │ ✓ Fits │ 1.12 GiB │ 2.00 B  │ CPU      │      4 │ 58.3     │ 14.2     │  98.4 │ ★★★★★        │
│ 2 │ Gemma 4 2B (ggml-org)        │ ✓ Fits │ 1.50 GiB │ 2.00 B  │ CPU      │      4 │ 55.1     │ 13.9     │  95.8 │ ★★★★★        │
│ 3 │ Gemma 4 4B (ggml-org)        │ ✓ Fits │ 4.95 GiB │ 7.52 B  │ CPU      │      4 │ 28.2     │  3.7     │  28.1 │ ★★           │
╰───┴──────────────────────────────┴────────┴──────────┴─────────┴──────────┴────────┴──────────┴──────────┴───────┴──────────────╯
```

## Requirements

- Python 3.11+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) built locally — specifically the `llama-bench` binary
- Internet access for the first run (models are downloaded from HuggingFace)

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

## Quick Start

```bash
# 1. See what hardware was detected and which models would be selected
llm-bench sysinfo

# 2. Run benchmarks (auto-detects hardware, picks model profile, caches results)
llm-bench run --llama-bench /path/to/llama-bench

# 3. Browse past runs
llm-bench results list
llm-bench results show 20240501-143022-abc123

# 4. Compare two runs (e.g. before/after adding RAM)
llm-bench results compare 20240501-143022-abc123 20240601-092011-def456
```

## How It Works

1. **Hardware detection** — reads CPU model/cores, total/available RAM, and GPU (NVIDIA via `nvidia-smi`, AMD via `rocm-smi`, Apple via `system_profiler`).
2. **Profile selection** — picks the best-fit YAML profile from `data/models/`:
   - `low_ram.yaml` — < 8 GB RAM
   - `medium_ram.yaml` — 8–16 GB RAM
   - `high_ram.yaml` — 16 GB+ RAM
3. **Cache lookup** — for each model, checks `~/.llm-bench/results/` for a prior run with the same hardware fingerprint + benchmark params. Hits are shown instantly with a `(cached)` badge.
4. **Benchmark** — runs `llama-bench -hf <repo> -p 512 -n 200 -r 5 -o json`, streams download/progress to the terminal, and parses the JSON output.
5. **Rating** — computes `score = 0.7 × (TG t/s ÷ best) + 0.3 × (PP t/s ÷ best)`, scaled 0–100 and shown as 1–5 stars. TG (token generation) is weighted more because it determines chat response speed.
6. **Save** — writes results to `~/.llm-bench/results/<run-id>/` for future comparisons.

## Custom Model Profiles

Place YAML files in `~/.llm-bench/models/` to override or extend the bundled profiles:

```yaml
profile: my_gpu_rig
description: "Custom models for my RTX 4090"
min_ram_gb: 32
max_ram_gb: 9999

models:
  - name: "Llama 3.3 70B"
    hf_repo: "ggml-org/Llama-3.3-70B-Instruct-GGUF"
    estimated_size_gb: 42.5
    description: "Meta's largest public model"
    tags: [large, llama]

  - name: "Qwen2.5 72B"
    hf_repo: "ggml-org/Qwen2.5-72B-Instruct-GGUF"
    estimated_size_gb: 43.0
    description: "Alibaba's 72B model"
    tags: [large, qwen]
```

Or pass a file directly at runtime:

```bash
llm-bench run --models-file ~/my-models.yaml
```

## CLI Reference

```
llm-bench run [OPTIONS] [EXTRA_LLAMA_ARGS]...

  --llama-bench PATH    Path to llama-bench binary
                        (default: /home/homelab/repos/llama.cpp/build/bin/llama-bench)
  --models-file FILE    Override auto-selected YAML profile
  --fresh               Ignore cached results; re-run everything
  -r, --repetitions N   Repetitions per test (default: 5)
  -p, --n-prompt N      Prompt token count for pp test (default: 512)
  -n, --n-gen N         Generation token count for tg test (default: 200)
  --hf-token TOKEN      HuggingFace token (or set HF_TOKEN env var)
  -t, --threads N       CPU thread count passed to llama-bench
  -o, --output FORMAT   table | json | markdown (default: table)

  Any extra options after -- are forwarded verbatim to llama-bench:
    llm-bench run -- -fa 1 -ngl 32

llm-bench sysinfo       Show hardware info and selected profile
llm-bench models list   List all available model profiles
llm-bench results list  List past runs
llm-bench results show RUN_ID
llm-bench results compare RUN_ID_A RUN_ID_B
```

## Result Storage

```
~/.llm-bench/
├── models/                   ← your custom YAML files (optional)
└── results/
    └── 20240501-143022-abc123/
        ├── meta.json         ← hw fingerprint, params, timestamp
        └── results.json      ← benchmark results per model
```

Results are **never mutated** — each run creates a new directory. The cache lookup is keyed on `SHA256(hw_fingerprint + params + hf_repo)`, so changing any parameter (or running on different hardware) triggers a fresh benchmark.

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
PYTHONPATH="" uv run pytest tests/test_parser.py -v                 # single file
PYTHONPATH="" uv run pytest tests/test_storage.py::test_find_cached_result -v  # single test
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
