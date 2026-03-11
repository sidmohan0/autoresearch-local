from __future__ import annotations

import json
import unittest
from pathlib import Path

from autoresearch_local.backends import (
    LlamaCppRunner,
    OllamaRunner,
    TESTED_LLAMA_CPP_BUILD,
    summarize_model_load_failure,
)


class BackendParsingTests(unittest.TestCase):
    def test_parse_llama_cli_output(self) -> None:
        fixture = Path(__file__).with_name("data").joinpath("llama_cli_output.txt").read_text(encoding="utf-8")
        run = LlamaCppRunner.parse_cli_output(fixture, elapsed_ms=2100.0)
        self.assertEqual(run.prompt_tokens, 160)
        self.assertEqual(run.decode_tokens, 96)
        self.assertAlmostEqual(run.prompt_tps, 200.0)
        self.assertAlmostEqual(run.decode_tps, 80.0)
        self.assertAlmostEqual(run.total_ms, 2050.0)

    def test_parse_ollama_generate_response(self) -> None:
        payload = {
            "total_duration": 2_500_000_000,
            "prompt_eval_duration": 1_000_000_000,
            "prompt_eval_count": 200,
            "eval_duration": 1_200_000_000,
            "eval_count": 120,
        }
        run = OllamaRunner.parse_generate_response(payload)
        self.assertAlmostEqual(run.total_ms, 2500.0)
        self.assertAlmostEqual(run.prompt_tps, 200.0)
        self.assertAlmostEqual(run.decode_tps, 100.0)

    def test_parse_compact_llama_cli_output(self) -> None:
        fixture = """
Hello!
[ Prompt: 105.4 t/s | Generation: 24.0 t/s ]
Exiting...
""".strip()
        run = LlamaCppRunner.parse_cli_output(fixture, elapsed_ms=3210.0)
        self.assertAlmostEqual(run.prompt_tps, 105.4)
        self.assertAlmostEqual(run.decode_tps, 24.0)
        self.assertAlmostEqual(run.total_ms, 3210.0)

    def test_summarize_model_load_failure_for_incomplete_model(self) -> None:
        model = Path("/tmp/example.gguf")
        output = """
ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
llama_model_load: error loading model: tensor 'blk.0.ffn_down.weight' data is not within the file bounds, model is corrupted or incomplete
llama_model_load_from_file_impl: failed to load model
""".strip()
        summary = summarize_model_load_failure(model, output)
        self.assertIn("The file appears incomplete or corrupted", summary)
        self.assertIn("not the root cause", summary)

    def test_compatibility_warning_for_non_tested_build(self) -> None:
        runner = object.__new__(LlamaCppRunner)
        runner.version = "build: something-else"
        warning = LlamaCppRunner.compatibility_warning(runner)
        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertIn(TESTED_LLAMA_CPP_BUILD, warning)

    def test_compatibility_warning_accepts_equivalent_version_format(self) -> None:
        runner = object.__new__(LlamaCppRunner)
        runner.version = "version: 8260 (96cfc4992)"
        warning = LlamaCppRunner.compatibility_warning(runner)
        self.assertIsNone(warning)


if __name__ == "__main__":
    unittest.main()
