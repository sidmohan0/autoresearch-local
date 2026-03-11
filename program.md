# autoresearch-local program

This repository applies the `autoresearch` structure to local inference tuning on Apple Silicon.

## Goal

Given a fixed `.gguf` model and a fixed benchmark harness, improve local inference performance without changing the model or cheating the benchmark.

The primary score is:

- lowest weighted end-to-end latency across the benchmark suite

Secondary metrics are observed and logged:

- prompt tokens/sec
- decode tokens/sec
- crash-free run rate

## In-Scope Files

Read these files before making changes:

- `README.md`
- `src/autoresearch_local/cli.py`
- `src/autoresearch_local/system.py`
- `src/autoresearch_local/prompts.py`
- `src/autoresearch_local/profiles.py`
- `src/autoresearch_local/backends.py`
- `src/autoresearch_local/tuning.py`

## Allowed Changes

You may change:

- benchmark orchestration
- tuning heuristics
- backend flag handling
- scoring and reporting
- docs and tests

You may not change:

- the benchmark prompt scenarios in a way that makes them easier just to improve the score
- the definition of the saved profile after a tuning run
- the benchmark comparison method without documenting the change

## Evaluation Loop

1. Establish a baseline with `llama.cpp` default invocation.
2. Run the fixed benchmark suite.
3. Log:
   - config label
   - weighted latency score
   - prompt tokens/sec
   - decode tokens/sec
   - success/failure
4. Try one tuning improvement at a time.
5. Keep only changes that measurably improve the benchmark or materially simplify the code.
6. If a change only improves one narrow metric while making end-to-end latency worse, reject it.

## Simplicity Rule

If two approaches benchmark similarly, keep the simpler one.

Small speed wins are not worth brittle parsing, backend-specific hacks without tests, or misleading output.

## Reporting Rule

Be precise in claims:

- say "better than the default invocation we measured"
- do not say "faster than llama.cpp" when the project itself uses `llama.cpp`
- treat `tokens/sec` as an observed metric, not the only objective
