from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autoresearch_local.system import candidate_model_roots, detect_backends, discover_models


class SystemDiscoveryTests(unittest.TestCase):
    def test_detect_backends_includes_brew_key(self) -> None:
        backends = detect_backends()
        self.assertIn("brew", backends)
        self.assertIn("version", backends["brew"].__dict__)

    def test_discover_models_finds_gguf_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models" / "tiny.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"gguf")
            with mock.patch.dict(os.environ, {"AUTORESEARCH_LOCAL_MODEL_DIRS": str(root / "models")}):
                discovered = discover_models(cwd=root, max_depth=2)
            paths = {item.path for item in discovered}
            self.assertIn(str(model.resolve()), paths)

    def test_candidate_model_roots_respects_env_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extra = root / "extra-models"
            extra.mkdir()
            with mock.patch.dict(os.environ, {"AUTORESEARCH_LOCAL_MODEL_DIRS": str(extra)}):
                roots = candidate_model_roots(cwd=root)
            self.assertIn(extra.resolve(), roots)


if __name__ == "__main__":
    unittest.main()
