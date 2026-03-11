from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .prompts import Scenario


def cache_dir() -> Path:
    override = os.environ.get("AUTORESEARCH_LOCAL_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cache" / "autoresearch-local"


def profiles_dir() -> Path:
    return cache_dir() / "profiles"


@dataclass(frozen=True)
class LlamaCppConfig:
    threads: int
    ctx_size: int
    batch_size: int
    ubatch_size: int
    gpu_layers: int = 999
    flash_attention: bool = True

    def label(self) -> str:
        flash = "fa" if self.flash_attention else "nofa"
        return (
            f"thr{self.threads}-ctx{self.ctx_size}-"
            f"b{self.batch_size}-ub{self.ubatch_size}-ngl{self.gpu_layers}-{flash}"
        )

    def to_ollama_options(self) -> dict[str, int | float]:
        return {
            "num_thread": self.threads,
            "num_ctx": self.ctx_size,
            "num_gpu": self.gpu_layers,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
        }


@dataclass(frozen=True)
class LlamaCppOverrides:
    threads: int | None = None
    ctx_size: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    gpu_layers: int | None = None
    flash_attention: bool | None = None


@dataclass(frozen=True)
class ScenarioAggregate:
    name: str
    weight: float
    median_total_ms: float
    median_prompt_tps: float
    median_decode_tps: float
    repeats: int


@dataclass(frozen=True)
class BenchmarkSummary:
    backend: str
    label: str
    score_ms: float
    prompt_tps: float
    decode_tps: float
    scenarios: list[ScenarioAggregate]


@dataclass(frozen=True)
class SavedProfile:
    model_path: str
    model_id: str
    model_size_bytes: int
    backend: str
    config: LlamaCppConfig
    benchmark: BenchmarkSummary


def ensure_cache_dirs() -> None:
    profiles_dir().mkdir(parents=True, exist_ok=True)


def model_fingerprint(model_path: Path) -> str:
    stat = model_path.stat()
    digest = hashlib.sha256()
    digest.update(str(model_path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()[:16]


def profile_path_for(model_path: Path) -> Path:
    return profiles_dir() / f"{model_fingerprint(model_path)}.json"


def save_profile(profile: SavedProfile) -> Path:
    ensure_cache_dirs()
    destination = profile_path_for(Path(profile.model_path))
    destination.write_text(json.dumps(asdict(profile), indent=2) + "\n", encoding="utf-8")
    return destination


def load_profile(model_path: Path) -> SavedProfile | None:
    path = profile_path_for(model_path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = LlamaCppConfig(**raw["config"])
    scenarios = [ScenarioAggregate(**item) for item in raw["benchmark"]["scenarios"]]
    benchmark = BenchmarkSummary(
        backend=raw["benchmark"]["backend"],
        label=raw["benchmark"]["label"],
        score_ms=raw["benchmark"]["score_ms"],
        prompt_tps=raw["benchmark"]["prompt_tps"],
        decode_tps=raw["benchmark"]["decode_tps"],
        scenarios=scenarios,
    )
    return SavedProfile(
        model_path=raw["model_path"],
        model_id=raw["model_id"],
        model_size_bytes=raw["model_size_bytes"],
        backend=raw["backend"],
        config=config,
        benchmark=benchmark,
    )


def scenario_to_dict(scenario: Scenario) -> dict[str, str | int | float]:
    return asdict(scenario)
