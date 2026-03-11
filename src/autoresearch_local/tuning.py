from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial
import re
from pathlib import Path
from statistics import median

from tqdm.auto import tqdm

from .backends import BackendError, BenchmarkRun, LlamaCppRunner, OllamaRunner
from .profiles import (
    BenchmarkSummary,
    LlamaCppConfig,
    LlamaCppOverrides,
    SavedProfile,
    ScenarioAggregate,
    model_fingerprint,
)
from .prompts import Scenario, default_scenarios
from .system import SystemProfile


DEFAULT_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class CandidateResult:
    config: LlamaCppConfig | None
    summary: BenchmarkSummary


@dataclass(frozen=True)
class AblationStep:
    label: str
    description: str
    overrides: LlamaCppOverrides | None
    summary: BenchmarkSummary


@dataclass(frozen=True)
class ShapleyContribution:
    label: str
    description: str
    contribution_ms: float
    contribution_percent_of_total: float


@dataclass(frozen=True)
class ShapleyAttribution:
    stock_summary: BenchmarkSummary
    full_summary: BenchmarkSummary
    subset_summaries: dict[tuple[str, ...], BenchmarkSummary]
    contributions: list[ShapleyContribution]


def improvement_percent(baseline_score_ms: float, candidate_score_ms: float) -> float:
    if baseline_score_ms <= 0:
        return 0.0
    return ((baseline_score_ms - candidate_score_ms) / baseline_score_ms) * 100.0


def _short_error_message(error: BackendError) -> str:
    first_line = str(error).splitlines()[0].strip()
    return first_line or "benchmark failed"


def heuristic_default_config(profile: SystemProfile, model_path: Path) -> LlamaCppConfig:
    del model_path
    perf = max(1, profile.performance_cores)
    if profile.memory_gb and profile.memory_gb <= 16:
        ctx_size = 4096
        batch_size = 512
        ubatch_size = 128
    else:
        ctx_size = 8192
        batch_size = 1024
        ubatch_size = 256
    return LlamaCppConfig(
        threads=perf,
        ctx_size=ctx_size,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        gpu_layers=999 if profile.is_apple_silicon else 0,
        flash_attention=True,
    )


def generate_candidate_configs(
    profile: SystemProfile,
    model_path: Path,
    max_candidates: int = 12,
) -> list[LlamaCppConfig]:
    baseline = heuristic_default_config(profile, model_path)
    perf = max(1, profile.performance_cores)
    threads = sorted({max(1, perf - 1), perf, min(profile.logical_cpu_count, perf + 2)})
    ctx_sizes = [2048, 4096] if profile.memory_gb <= 16 else [4096, 8192]
    batch_pairs = [(256, 64), (512, 128), (1024, 256)]
    flash_values = [True, False]
    candidates: list[LlamaCppConfig] = []
    seen: set[tuple[int, int, int, int, int, bool]] = set()
    preferred_pairs = [(baseline.batch_size, baseline.ubatch_size)] + batch_pairs
    for thread_count in threads:
        for ctx_size in ctx_sizes:
            for batch_size, ubatch_size in preferred_pairs:
                if ubatch_size > batch_size:
                    continue
                for flash_attention in flash_values:
                    candidate = LlamaCppConfig(
                        threads=thread_count,
                        ctx_size=ctx_size,
                        batch_size=batch_size,
                        ubatch_size=ubatch_size,
                        gpu_layers=baseline.gpu_layers,
                        flash_attention=flash_attention,
                    )
                    key = (
                        candidate.threads,
                        candidate.ctx_size,
                        candidate.batch_size,
                        candidate.ubatch_size,
                        candidate.gpu_layers,
                        candidate.flash_attention,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(candidate)
    candidates.sort(key=lambda item: (item.ctx_size, item.threads, item.batch_size, item.ubatch_size, not item.flash_attention))
    if baseline not in candidates:
        candidates.insert(0, baseline)
    return candidates[:max_candidates]


def _aggregate_runs(name: str, weight: float, runs: list[BenchmarkRun]) -> ScenarioAggregate:
    return ScenarioAggregate(
        name=name,
        weight=weight,
        median_total_ms=float(median([run.total_ms for run in runs])),
        median_prompt_tps=float(median([run.prompt_tps for run in runs])),
        median_decode_tps=float(median([run.decode_tps for run in runs])),
        repeats=len(runs),
    )


def _build_summary(backend: str, label: str, aggregates: list[ScenarioAggregate]) -> BenchmarkSummary:
    score_ms = sum(item.median_total_ms * item.weight for item in aggregates)
    total_weight = sum(item.weight for item in aggregates) or 1.0
    prompt_tps = sum(item.median_prompt_tps * item.weight for item in aggregates) / total_weight
    decode_tps = sum(item.median_decode_tps * item.weight for item in aggregates) / total_weight
    return BenchmarkSummary(
        backend=backend,
        label=label,
        score_ms=score_ms,
        prompt_tps=prompt_tps,
        decode_tps=decode_tps,
        scenarios=aggregates,
    )


def benchmark_llama_cpp(
    runner: LlamaCppRunner,
    model_path: Path,
    scenarios: list[Scenario] | None = None,
    config: LlamaCppConfig | LlamaCppOverrides | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    label: str | None = None,
    show_progress: bool = True,
) -> BenchmarkSummary:
    scenario_list = scenarios or default_scenarios()
    benchmark_label = label or (config.label() if config else "default")
    aggregates: list[ScenarioAggregate] = []
    total_runs = sum(scenario.repeats for scenario in scenario_list)
    progress = tqdm(
        total=total_runs,
        desc=f"benchmark {benchmark_label}",
        leave=False,
        disable=not show_progress,
    )
    for scenario in scenario_list:
        runs: list[BenchmarkRun] = []
        for _ in range(scenario.repeats):
            progress.set_postfix_str(f"{scenario.name} ({scenario.max_tokens} tok)")
            runs.append(
                runner.run_once(
                    model_path=model_path,
                    prompt=scenario.prompt,
                    max_tokens=scenario.max_tokens,
                    config=config,
                    timeout_seconds=timeout_seconds,
                )
            )
            progress.update(1)
        aggregates.append(_aggregate_runs(scenario.name, scenario.weight, runs))
    progress.close()
    return _build_summary("llama.cpp", benchmark_label, aggregates)


def benchmark_ollama(
    runner: OllamaRunner,
    model_path: Path,
    config: LlamaCppConfig | None = None,
    scenarios: list[Scenario] | None = None,
    show_progress: bool = True,
) -> BenchmarkSummary:
    scenario_list = scenarios or default_scenarios()
    aggregates: list[ScenarioAggregate] = []
    stem = re.sub(r"[^a-z0-9]+", "-", model_path.stem.lower()).strip("-")
    model_name = runner.ensure_model(model_path, f"autoresearch-local-{stem}")
    total_runs = sum(scenario.repeats for scenario in scenario_list)
    progress = tqdm(
        total=total_runs,
        desc="benchmark ollama",
        leave=False,
        disable=not show_progress,
    )
    for scenario in scenario_list:
        runs: list[BenchmarkRun] = []
        for _ in range(scenario.repeats):
            progress.set_postfix_str(f"{scenario.name} ({scenario.max_tokens} tok)")
            runs.append(
                runner.run_once(
                    model_name=model_name,
                    prompt=scenario.prompt,
                    max_tokens=scenario.max_tokens,
                    config=config,
                )
            )
            progress.update(1)
        aggregates.append(_aggregate_runs(scenario.name, scenario.weight, runs))
    progress.close()
    return _build_summary("ollama", "same-gguf-wrapper", aggregates)


def _merge_overrides(*overrides: LlamaCppOverrides) -> LlamaCppOverrides:
    merged: dict[str, object] = {}
    for override in overrides:
        for field in ("threads", "ctx_size", "batch_size", "ubatch_size", "gpu_layers", "flash_attention"):
            value = getattr(override, field)
            if value is not None:
                merged[field] = value
    return LlamaCppOverrides(**merged)


def _subset_key(labels: tuple[str, ...], order: list[str]) -> tuple[str, ...]:
    return tuple(sorted(labels, key=order.index))


def _subset_label(labels: tuple[str, ...]) -> str:
    return "stock" if not labels else "subset-" + "-".join(labels)


def _subset_overrides(
    plan: list[tuple[str, str, LlamaCppOverrides]],
    labels: tuple[str, ...],
) -> LlamaCppOverrides | None:
    if not labels:
        return None
    overrides_by_label = {label: override for label, _, override in plan}
    selected = [overrides_by_label[label] for label in labels]
    return _merge_overrides(*selected)


def _all_subset_keys(order: list[str]) -> list[tuple[str, ...]]:
    keys: list[tuple[str, ...]] = []
    for size in range(len(order) + 1):
        for subset in combinations(order, size):
            keys.append(_subset_key(subset, order))
    return keys


def compute_shapley_values(
    order: list[str],
    values: dict[tuple[str, ...], float],
) -> dict[str, float]:
    total_features = len(order)
    denominator = factorial(total_features) or 1
    contributions: dict[str, float] = {}
    for feature in order:
        others = [label for label in order if label != feature]
        contribution = 0.0
        for size in range(len(others) + 1):
            for subset in combinations(others, size):
                subset_key = _subset_key(subset, order)
                with_feature_key = _subset_key(subset + (feature,), order)
                weight = factorial(size) * factorial(total_features - size - 1) / denominator
                contribution += weight * (values[with_feature_key] - values[subset_key])
        contributions[feature] = contribution
    return contributions


def ablation_plan(tuned_config: LlamaCppConfig) -> list[tuple[str, str, LlamaCppOverrides]]:
    steps = [
        (
            "ctx",
            f"apply tuned context size ({tuned_config.ctx_size})",
            LlamaCppOverrides(ctx_size=tuned_config.ctx_size),
        ),
        (
            "batching",
            f"apply tuned batch sizes (batch={tuned_config.batch_size}, ubatch={tuned_config.ubatch_size})",
            LlamaCppOverrides(batch_size=tuned_config.batch_size, ubatch_size=tuned_config.ubatch_size),
        ),
        (
            "threads",
            f"apply tuned CPU threads ({tuned_config.threads})",
            LlamaCppOverrides(threads=tuned_config.threads),
        ),
        (
            "gpu_layers",
            f"apply tuned GPU offload ({tuned_config.gpu_layers})",
            LlamaCppOverrides(gpu_layers=tuned_config.gpu_layers),
        ),
        (
            "flash_attention",
            f"apply tuned flash attention ({'on' if tuned_config.flash_attention else 'off'})",
            LlamaCppOverrides(flash_attention=tuned_config.flash_attention),
        ),
    ]
    return steps


def run_ablation(
    model_path: Path,
    tuned_config: LlamaCppConfig,
    runner: LlamaCppRunner | None = None,
) -> list[AblationStep]:
    llama_runner = runner or LlamaCppRunner()
    steps: list[AblationStep] = []
    baseline = benchmark_llama_cpp(llama_runner, model_path, config=None, label="stock")
    steps.append(AblationStep("stock", "stock llama.cpp defaults", None, baseline))
    cumulative = LlamaCppOverrides()
    plan = ablation_plan(tuned_config)
    progress = tqdm(plan, desc="ablation steps")
    for label, description, override in progress:
        cumulative = _merge_overrides(cumulative, override)
        summary = benchmark_llama_cpp(llama_runner, model_path, config=cumulative, label=label)
        steps.append(AblationStep(label, description, cumulative, summary))
        progress.set_postfix(step=label, score=f"{summary.score_ms:.1f}ms")
    progress.close()
    return steps


def run_shapley(
    model_path: Path,
    tuned_config: LlamaCppConfig,
    runner: LlamaCppRunner | None = None,
) -> ShapleyAttribution:
    llama_runner = runner or LlamaCppRunner()
    plan = ablation_plan(tuned_config)
    order = [label for label, _, _ in plan]
    descriptions = {label: description for label, description, _ in plan}
    subset_summaries: dict[tuple[str, ...], BenchmarkSummary] = {}
    subset_keys = _all_subset_keys(order)
    progress = tqdm(subset_keys, desc="shapley subsets")
    for subset_key in progress:
        overrides = _subset_overrides(plan, subset_key)
        summary = benchmark_llama_cpp(
            llama_runner,
            model_path,
            config=overrides,
            label=_subset_label(subset_key),
        )
        subset_summaries[subset_key] = summary
        progress.set_postfix(subset=_subset_label(subset_key), score=f"{summary.score_ms:.1f}ms")
    progress.close()
    stock_summary = subset_summaries[()]
    full_key = tuple(order)
    full_summary = subset_summaries[full_key]
    values = {
        key: stock_summary.score_ms - summary.score_ms
        for key, summary in subset_summaries.items()
    }
    raw_contributions = compute_shapley_values(order, values)
    total_improvement_ms = stock_summary.score_ms - full_summary.score_ms
    contributions = [
        ShapleyContribution(
            label=label,
            description=descriptions[label],
            contribution_ms=raw_contributions[label],
            contribution_percent_of_total=(
                (raw_contributions[label] / total_improvement_ms) * 100.0 if total_improvement_ms > 0 else 0.0
            ),
        )
        for label in order
    ]
    return ShapleyAttribution(
        stock_summary=stock_summary,
        full_summary=full_summary,
        subset_summaries=subset_summaries,
        contributions=contributions,
    )


def tune_model(
    profile: SystemProfile,
    model_path: Path,
    max_candidates: int = 12,
) -> tuple[SavedProfile, BenchmarkSummary]:
    runner = LlamaCppRunner()
    default_summary = benchmark_llama_cpp(runner, model_path, config=None, label="default")
    best_config: LlamaCppConfig | None = None
    best_summary = default_summary
    candidates = generate_candidate_configs(profile, model_path, max_candidates=max_candidates)
    candidate_progress = tqdm(candidates, desc="tuning candidates")
    for candidate in candidate_progress:
        try:
            summary = benchmark_llama_cpp(runner, model_path, config=candidate)
        except BackendError as error:
            tqdm.write(f"skip {candidate.label()}: {_short_error_message(error)}")
            continue
        if summary.score_ms < best_summary.score_ms:
            best_config = candidate
            best_summary = summary
        candidate_progress.set_postfix(
            current=candidate.label(),
            best=best_summary.label,
            score=f"{best_summary.score_ms:.1f}ms",
        )
    candidate_progress.close()
    selected_config = best_config or heuristic_default_config(profile, model_path)
    saved = SavedProfile(
        model_path=str(model_path.resolve()),
        model_id=model_fingerprint(model_path),
        model_size_bytes=model_path.stat().st_size,
        backend="llama.cpp",
        config=selected_config,
        benchmark=best_summary,
    )
    return saved, default_summary


def format_summary_table(summaries: list[BenchmarkSummary]) -> str:
    lines = ["label\tbackend\tscore_ms\tprompt_tps\tdecode_tps"]
    for summary in summaries:
        lines.append(
            "\t".join(
                [
                    summary.label,
                    summary.backend,
                    f"{summary.score_ms:.1f}",
                    f"{summary.prompt_tps:.1f}",
                    f"{summary.decode_tps:.1f}",
                ]
            )
        )
    return "\n".join(lines)


def format_ablation_table(steps: list[AblationStep]) -> str:
    lines = ["step\tscore_ms\tprompt_tps\tdecode_tps\tdelta_vs_prev_ms\tdelta_vs_stock_pct\tdescription"]
    stock_score = steps[0].summary.score_ms if steps else 0.0
    previous_score = stock_score
    for step in steps:
        delta_prev = step.summary.score_ms - previous_score if step.label != "stock" else 0.0
        delta_stock_pct = improvement_percent(stock_score, step.summary.score_ms)
        lines.append(
            "\t".join(
                [
                    step.label,
                    f"{step.summary.score_ms:.1f}",
                    f"{step.summary.prompt_tps:.1f}",
                    f"{step.summary.decode_tps:.1f}",
                    f"{delta_prev:.1f}",
                    f"{delta_stock_pct:.1f}",
                    step.description,
                ]
            )
        )
        previous_score = step.summary.score_ms
    return "\n".join(lines)


def format_shapley_table(attribution: ShapleyAttribution) -> str:
    lines = ["factor\tcontribution_ms\tshare_of_total_pct\tdescription"]
    for contribution in attribution.contributions:
        lines.append(
            "\t".join(
                [
                    contribution.label,
                    f"{contribution.contribution_ms:.1f}",
                    f"{contribution.contribution_percent_of_total:.1f}",
                    contribution.description,
                ]
            )
        )
    return "\n".join(lines)
