from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autoresearch_local.tuning import (
    _all_subset_keys,
    ablation_plan,
    compute_shapley_values,
    generate_candidate_configs,
    heuristic_default_config,
    improvement_percent,
)
from autoresearch_local.profiles import LlamaCppConfig
from autoresearch_local.system import SystemProfile


class TuningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = SystemProfile(
            platform="Darwin",
            machine="arm64",
            processor="arm",
            cpu_brand="Apple M3",
            memory_gb=16.0,
            logical_cpu_count=8,
            performance_cores=4,
        )

    def test_heuristic_default_config_targets_apple_silicon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"1")
            config = heuristic_default_config(self.profile, model)
            self.assertEqual(config.gpu_layers, 999)
            self.assertEqual(config.ctx_size, 4096)

    def test_generate_candidate_configs_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"1")
            configs = generate_candidate_configs(self.profile, model, max_candidates=8)
            self.assertLessEqual(len(configs), 8)
            labels = {config.label() for config in configs}
            self.assertEqual(len(labels), len(configs))

    def test_improvement_percent(self) -> None:
        self.assertAlmostEqual(improvement_percent(100.0, 75.0), 25.0)

    def test_ablation_plan_contains_expected_steps(self) -> None:
        plan = ablation_plan(
            LlamaCppConfig(
                threads=4,
                ctx_size=2048,
                batch_size=512,
                ubatch_size=128,
                gpu_layers=999,
                flash_attention=True,
            )
        )
        labels = [label for label, _, _ in plan]
        self.assertEqual(labels, ["ctx", "batching", "threads", "gpu_layers", "flash_attention"])

    def test_all_subset_keys_for_five_features(self) -> None:
        order = ["ctx", "batching", "threads", "gpu_layers", "flash_attention"]
        keys = _all_subset_keys(order)
        self.assertEqual(len(keys), 32)
        self.assertIn((), keys)
        self.assertIn(tuple(order), keys)

    def test_compute_shapley_values_for_additive_game(self) -> None:
        order = ["ctx", "batching", "threads"]
        weights = {"ctx": 100.0, "batching": 40.0, "threads": -5.0}
        values = {}
        for subset in _all_subset_keys(order):
            values[subset] = sum(weights[label] for label in subset)
        contributions = compute_shapley_values(order, values)
        self.assertAlmostEqual(contributions["ctx"], 100.0)
        self.assertAlmostEqual(contributions["batching"], 40.0)
        self.assertAlmostEqual(contributions["threads"], -5.0)
        self.assertAlmostEqual(sum(contributions.values()), values[tuple(order)])


if __name__ == "__main__":
    unittest.main()
