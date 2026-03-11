from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autoresearch_local.profiles import (
    BenchmarkSummary,
    LlamaCppConfig,
    SavedProfile,
    ScenarioAggregate,
    load_profile,
    model_fingerprint,
    save_profile,
)


class ProfileStorageTests(unittest.TestCase):
    def test_fingerprint_changes_with_file_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.write_bytes(b"hello")
            first = model_fingerprint(model_path)
            model_path.write_bytes(b"hello world")
            second = model_fingerprint(model_path)
            self.assertNotEqual(first, second)

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.write_bytes(b"hello")
            profile = SavedProfile(
                model_path=str(model_path),
                model_id=model_fingerprint(model_path),
                model_size_bytes=model_path.stat().st_size,
                backend="llama.cpp",
                config=LlamaCppConfig(
                    threads=4,
                    ctx_size=4096,
                    batch_size=512,
                    ubatch_size=128,
                ),
                benchmark=BenchmarkSummary(
                    backend="llama.cpp",
                    label="test",
                    score_ms=123.4,
                    prompt_tps=111.1,
                    decode_tps=99.9,
                    scenarios=[
                        ScenarioAggregate(
                            name="short",
                            weight=1.0,
                            median_total_ms=123.4,
                            median_prompt_tps=111.1,
                            median_decode_tps=99.9,
                            repeats=2,
                        )
                    ],
                ),
            )
            with mock.patch.dict("os.environ", {"AUTORESEARCH_LOCAL_CACHE_DIR": directory}):
                path = save_profile(profile)
                loaded = load_profile(model_path)
                self.assertEqual(path.name, f"{profile.model_id}.json")
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(loaded.config.batch_size, 512)
                self.assertEqual(loaded.benchmark.label, "test")


if __name__ == "__main__":
    unittest.main()
