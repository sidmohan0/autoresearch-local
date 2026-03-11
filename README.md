# autoresearch-local

`autoresearch-local` is an Apple Silicon-focused GGUF tuning and serving tool built from the structure of Karpathy's `autoresearch`.

This fork does **not** claim to be a new inference engine. It is an orchestration layer around existing local runtimes:

- `llama.cpp` for direct GGUF loading and serving
- `ollama` for product-level comparison against the same model

The goal is simple: given a `.gguf` file on a Mac, benchmark a fixed set of prompts, tune the runtime knobs that materially affect local performance, save the best profile, and launch a local server with that profile.

## Versioning

This project is currently tested against `llama.cpp` build `b8260-96cfc4992`.

That matters because `autoresearch-local` depends on:

- the `llama-cli` flag surface
- the `llama-server` flag surface
- the benchmark timing output format

Different `llama.cpp` builds may still work, but they can change CLI behavior in ways that affect tuning and parsing. The tool reports detected backend versions in `inspect` and warns at runtime when the installed `llama.cpp` build differs from the tested build.

## What It Optimizes

Raw decode `tokens/sec` is not the only objective because it is easy to game and often ignores user-visible latency. This project uses a fixed benchmark harness and optimizes for:

- weighted end-to-end latency across short, medium, and long prompts
- prompt processing throughput
- decode throughput
- crash-free completion across all scenarios

The score is intentionally conservative: lower weighted latency wins. Throughput is reported, but it is not the only metric.

## Scope and Accuracy

This project is intended for:

- macOS on Apple Silicon
- local `.gguf` models
- `llama.cpp` tuning and serving
- `ollama` comparison runs against the same model

This project is **not**:

- a training harness
- a replacement for `llama.cpp`
- a claim that it outperforms `llama.cpp` kernels

If a tuned profile beats a default `llama.cpp` invocation or an Ollama wrapper on a given machine, that means the tuning strategy was better for that setup. It does **not** mean this repo invented a faster inference backend.

## Requirements

- macOS on Apple Silicon
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) installed and available on `PATH`
- optional: [Ollama](https://ollama.com/) for comparison runs

Install `llama.cpp` with Homebrew:

```bash
brew install llama.cpp
```

## Installation

```bash
git clone https://github.com/sidmohan0/autoresearch.git
cd autoresearch
uv sync
```

## Quick Start

Inspect the machine and backend availability:

```bash
uv run autoresearch-local inspect
```

If `llama.cpp` is missing, get install instructions:

```bash
uv run autoresearch-local setup
```

Or install it directly through the tool:

```bash
uv run autoresearch-local setup --install-llama-cpp
```

Tune a GGUF model:

```bash
uv run autoresearch-local tune /absolute/path/to/model.gguf
```

Benchmark the tuned profile against a default `llama.cpp` run and, optionally, Ollama:

```bash
uv run autoresearch-local benchmark /absolute/path/to/model.gguf --include-ollama
```

When `--include-ollama` is used, the tool imports the GGUF into Ollama once under a stable `autoresearch-local-...` model name and then reuses that model for later comparisons. The one-time import cost is intentionally kept out of the benchmark score.

Launch an OpenAI-compatible `llama-server` using the saved tuned profile:

```bash
uv run autoresearch-local serve /absolute/path/to/model.gguf --port 8080
```

Profiles are stored under `~/.cache/autoresearch-local/profiles/`.

## Example Result

Measured on March 10, 2026 on an Apple M3 Mac with 16 GB unified memory, using `llama.cpp` build `b8260-96cfc4992` and `Qwen2.5-7B-Instruct-Q4_K_M.gguf`.

Command:

```bash
uv run autoresearch-local tune /absolute/path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

Observed result:

| label | backend | score_ms | prompt_tps | decode_tps |
| --- | --- | ---: | ---: | ---: |
| default | llama.cpp | 26092.2 | 174.2 | 16.0 |
| thr3-ctx2048-b256-ub64-ngl999-fa | llama.cpp | 19634.5 | 104.7 | 7.3 |

This run produced `24.7%` lower weighted latency than the stock `llama.cpp` invocation for the fixed benchmark harness. The winning profile used:

- `threads=3`
- `ctx_size=2048`
- `batch_size=256`
- `ubatch_size=64`
- `gpu_layers=999`
- `flash_attention=on`

This should be read as a machine-specific benchmark result, not a general claim that these settings are best for all Apple Silicon systems or all GGUF models.

## CLI

### `inspect`

Prints machine information, backend availability, and discovered `.gguf` model paths from likely local model directories.

```bash
uv run autoresearch-local inspect --json
```

### `setup`

Explains what local runtime pieces are missing and can install `llama.cpp` via Homebrew on macOS.

```bash
uv run autoresearch-local setup
uv run autoresearch-local setup --install-llama-cpp
```

### `tune`

Runs a fixed benchmark suite against a generated candidate set of `llama.cpp` configurations, then saves the best-performing profile.

```bash
uv run autoresearch-local tune /path/to/model.gguf --repeats 2 --max-candidates 10
```

Long-running benchmark commands show `tqdm` progress bars for scenario repeats and candidate search.

What changes between candidates:

- CPU threads
- context size
- batch size
- micro-batch size
- Flash Attention toggle when supported by the installed `llama.cpp`
- GPU layer offload setting

What does not change:

- model file
- benchmark prompts
- decode length per scenario
- sampling seed and deterministic settings

### `benchmark`

Runs comparison benchmarks using:

- `llama.cpp` default invocation
- saved tuned `llama.cpp` profile
- optional Ollama model created from the same GGUF

```bash
uv run autoresearch-local benchmark /path/to/model.gguf --include-ollama --json
```

For a publishable benchmark artifact:

```bash
uv run autoresearch-local benchmark /path/to/model.gguf --export-json ./benchmark-report.json
```

### `ablate`

Runs a cumulative stock-to-tuned ablation so you can explain which groups of runtime changes produced the gain.

```bash
uv run autoresearch-local ablate /path/to/model.gguf
uv run autoresearch-local ablate /path/to/model.gguf --export-json ./ablation-report.json
```

### `shapley`

Runs an exact Shapley attribution over the same grouped runtime knobs used by `ablate`. This is slower than cumulative ablation because it benchmarks every subset once, but it gives order-independent attributions that sum to the measured stock-to-tuned improvement.

```bash
uv run autoresearch-local shapley /path/to/model.gguf
uv run autoresearch-local shapley /path/to/model.gguf --export-json ./shapley-report.json
```

### `serve`

Starts `llama-server` with the saved tuned profile. If no saved profile exists, it falls back to a heuristic default profile for the current machine.

```bash
uv run autoresearch-local serve /path/to/model.gguf --host 127.0.0.1 --port 8080
```

## Benchmark Methodology

The benchmark harness is fixed and measurable:

- short prompt, short completion
- medium prompt, medium completion
- long prompt, medium completion
- deterministic decode settings
- repeated runs per scenario
- weighted median latency aggregation

The selected profile is the one with the lowest weighted total latency across the suite. Prompt and decode throughput are also reported for interpretability.

This project does not currently measure GPU memory directly in a portable way across all supported backends. Failures due to memory pressure are surfaced as failed benchmark runs.

## Local Development

Run tests:

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

## Project Layout

```text
src/autoresearch_local/
  cli.py          command-line entrypoint
  system.py       machine and backend detection
  prompts.py      fixed benchmark prompt scenarios
  profiles.py     config/result models and profile persistence
  backends.py     llama.cpp and Ollama integrations
  tuning.py       candidate generation, benchmarking, tuning logic
tests/
  test_*.py       parser, config, and storage tests
program.md        agent loop for autonomous inference tuning work
```

## Origin

This repository started as a fork of Karpathy's `autoresearch`. The original training-oriented files remain in the repo history, but the installable project in this fork is now focused on Apple Silicon local inference tuning for GGUF models.
