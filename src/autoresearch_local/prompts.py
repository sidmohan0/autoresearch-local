from __future__ import annotations

from dataclasses import dataclass


_BASE_TEXT = (
    "Local language model inference on Apple Silicon benefits from careful runtime tuning. "
    "The right batch size, context length, and thread count can shift latency substantially. "
    "These prompts are intentionally deterministic and reused across every benchmark run so "
    "that candidate configurations are compared against the same workload."
)


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt: str
    max_tokens: int
    weight: float
    repeats: int = 2


def _repeat_text(multiplier: int) -> str:
    return " ".join([_BASE_TEXT] * multiplier)


def default_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="short",
            prompt=_repeat_text(6),
            max_tokens=96,
            weight=0.50,
            repeats=2,
        ),
        Scenario(
            name="medium",
            prompt=_repeat_text(30),
            max_tokens=160,
            weight=0.35,
            repeats=2,
        ),
        Scenario(
            name="long",
            prompt=_repeat_text(100),
            max_tokens=192,
            weight=0.15,
            repeats=2,
        ),
    ]
