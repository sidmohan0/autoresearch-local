from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path

from .backends import BackendError, LlamaCppRunner, OllamaRunner
from .profiles import load_profile, save_profile
from .system import (
    detect_backends,
    detect_system_profile,
    discover_models,
    format_inspect_json,
    format_inspect_text,
)
from .tuning import (
    AblationStep,
    ablation_plan,
    benchmark_llama_cpp,
    benchmark_ollama,
    format_ablation_table,
    format_shapley_table,
    format_summary_table,
    heuristic_default_config,
    improvement_percent,
    run_ablation,
    run_shapley,
    tune_model,
)


def _existing_model_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"model does not exist: {raw}")
    if path.suffix.lower() != ".gguf":
        raise argparse.ArgumentTypeError(f"expected a .gguf file: {raw}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch-local")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="show local machine and backend availability")
    inspect_parser.add_argument("--json", action="store_true", help="print machine information as JSON")

    setup_parser = subparsers.add_parser("setup", help="check or install required local backends")
    setup_parser.add_argument(
        "--install-llama-cpp",
        action="store_true",
        help="install llama.cpp with Homebrew if it is missing",
    )

    tune_parser = subparsers.add_parser("tune", help="benchmark candidate llama.cpp configs and save the best profile")
    tune_parser.add_argument("model", type=_existing_model_path)
    tune_parser.add_argument("--max-candidates", type=int, default=10)
    tune_parser.add_argument("--json", action="store_true", help="print saved profile as JSON")

    bench_parser = subparsers.add_parser("benchmark", help="compare default, tuned, and optional Ollama runs")
    bench_parser.add_argument("model", type=_existing_model_path)
    bench_parser.add_argument("--include-ollama", action="store_true")
    bench_parser.add_argument("--json", action="store_true", help="print summaries as JSON")
    bench_parser.add_argument("--export-json", type=Path, help="write a benchmark report JSON file")

    ablate_parser = subparsers.add_parser("ablate", help="run cumulative stock-to-tuned ablation analysis")
    ablate_parser.add_argument("model", type=_existing_model_path)
    ablate_parser.add_argument("--json", action="store_true", help="print ablation report as JSON")
    ablate_parser.add_argument("--export-json", type=Path, help="write an ablation report JSON file")

    shapley_parser = subparsers.add_parser("shapley", help="run exact Shapley attribution over tuned runtime knobs")
    shapley_parser.add_argument("model", type=_existing_model_path)
    shapley_parser.add_argument("--json", action="store_true", help="print Shapley attribution report as JSON")
    shapley_parser.add_argument("--export-json", type=Path, help="write a Shapley attribution report JSON file")

    serve_parser = subparsers.add_parser("serve", help="launch llama-server with the tuned profile")
    serve_parser.add_argument("model", type=_existing_model_path)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--print-command", action="store_true", help="print the command instead of executing it")

    return parser


def _profile_json(profile) -> str:
    return json.dumps(
        {
            "model_path": profile.model_path,
            "model_id": profile.model_id,
            "model_size_bytes": profile.model_size_bytes,
            "backend": profile.backend,
            "config": profile.config.__dict__,
            "benchmark": {
                "backend": profile.benchmark.backend,
                "label": profile.benchmark.label,
                "score_ms": profile.benchmark.score_ms,
                "prompt_tps": profile.benchmark.prompt_tps,
                "decode_tps": profile.benchmark.decode_tps,
                "scenarios": [item.__dict__ for item in profile.benchmark.scenarios],
            },
        },
        indent=2,
    )


def _summary_to_dict(summary) -> dict:
    return {
        "backend": summary.backend,
        "label": summary.label,
        "score_ms": summary.score_ms,
        "prompt_tps": summary.prompt_tps,
        "decode_tps": summary.decode_tps,
        "scenarios": [item.__dict__ for item in summary.scenarios],
    }


def _ablation_step_to_dict(step: AblationStep, stock_score_ms: float, previous_score_ms: float) -> dict:
    return {
        "label": step.label,
        "description": step.description,
        "score_ms": step.summary.score_ms,
        "prompt_tps": step.summary.prompt_tps,
        "decode_tps": step.summary.decode_tps,
        "delta_vs_previous_ms": step.summary.score_ms - previous_score_ms if step.label != "stock" else 0.0,
        "improvement_vs_stock_percent": improvement_percent(stock_score_ms, step.summary.score_ms),
        "summary": _summary_to_dict(step.summary),
    }


def _repro_footer(profile, model: Path, runner: LlamaCppRunner) -> str:
    llama_version = runner.version or "unknown"
    return (
        f"tested_with: model={model} | machine={profile.cpu_brand} | "
        f"memory_gb={profile.memory_gb:.1f} | llama.cpp={llama_version}"
    )


def _shapley_contribution_to_dict(contribution) -> dict:
    return {
        "label": contribution.label,
        "description": contribution.description,
        "contribution_ms": contribution.contribution_ms,
        "contribution_percent_of_total": contribution.contribution_percent_of_total,
    }


def _benchmark_report_payload(model: Path, system_profile, runner: LlamaCppRunner, summaries: list) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model),
        "system": {
            "platform": system_profile.platform,
            "machine": system_profile.machine,
            "cpu_brand": system_profile.cpu_brand,
            "memory_gb": system_profile.memory_gb,
            "logical_cpu_count": system_profile.logical_cpu_count,
            "performance_cores": system_profile.performance_cores,
        },
        "llama_cpp_version": runner.version,
        "summaries": [_summary_to_dict(summary) for summary in summaries],
    }


def _write_json_report(destination: Path, payload: dict) -> Path:
    target = destination.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def _resolve_tuned_config(model: Path, profile) -> tuple[object, bool]:
    saved_profile = load_profile(model)
    if saved_profile is not None:
        return saved_profile.config, True
    return heuristic_default_config(profile, model), False


def command_inspect(as_json: bool) -> int:
    profile = detect_system_profile()
    backends = detect_backends()
    models = discover_models()
    if as_json:
        print(format_inspect_json(profile, backends, models))
    else:
        print(format_inspect_text(profile, backends, models))
    return 0


def command_setup(install_llama_cpp: bool) -> int:
    backends = detect_backends()
    if backends["llama-cli"].available and backends["llama-server"].available:
        print("llama.cpp is already installed and available on PATH.")
        if backends["llama-cli"].version:
            print(f"llama-cli version: {backends['llama-cli'].version}")
        return 0

    brew = backends.get("brew")
    if install_llama_cpp:
        if brew is None or not brew.available:
            print("error: Homebrew is not installed. Install Homebrew first, then run `brew install llama.cpp`.", file=sys.stderr)
            return 1
        completed = subprocess.run([brew.executable or "brew", "install", "llama.cpp"])
        if completed.returncode != 0:
            return completed.returncode
        refreshed = detect_backends()
        if refreshed["llama-cli"].available and refreshed["llama-server"].available:
            print("Installed llama.cpp successfully.")
            return 0
        print(
            "error: `brew install llama.cpp` completed but `llama-cli`/`llama-server` are still not on PATH.",
            file=sys.stderr,
        )
        return 1

    print("llama.cpp is not installed.")
    if brew is not None and brew.available:
        print("Install it with:")
        print("  autoresearch-local setup --install-llama-cpp")
        print("or:")
        print("  brew install llama.cpp")
        return 0
    print("Homebrew is not installed. Install Homebrew first, then run `brew install llama.cpp`.")
    return 1


def command_tune(model: Path, max_candidates: int, as_json: bool) -> int:
    runner = LlamaCppRunner()
    warning = runner.compatibility_warning()
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    profile = detect_system_profile()
    saved_profile, default_summary = tune_model(profile, model, max_candidates=max_candidates)
    destination = save_profile(saved_profile)
    if as_json:
        print(_profile_json(saved_profile))
    else:
        summaries = [default_summary, saved_profile.benchmark]
        print(format_summary_table(summaries))
        improvement = improvement_percent(default_summary.score_ms, saved_profile.benchmark.score_ms)
        print(f"improvement_vs_default: {improvement:.1f}% lower weighted latency")
        print(_repro_footer(profile, model, runner))
        print(f"saved_profile: {destination}")
    return 0


def command_benchmark(model: Path, include_ollama: bool, as_json: bool, export_json: Path | None) -> int:
    profile = detect_system_profile()
    runner = LlamaCppRunner()
    warning = runner.compatibility_warning()
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    summaries = [benchmark_llama_cpp(runner, model, config=None, label="default")]
    tuned_config, _ = _resolve_tuned_config(model, profile)
    summaries.append(benchmark_llama_cpp(runner, model, config=tuned_config, label="tuned"))
    if include_ollama:
        summaries.append(benchmark_ollama(OllamaRunner(), model, config=tuned_config))
    report_payload = _benchmark_report_payload(model, profile, runner, summaries)
    exported_path: Path | None = None
    if export_json is not None:
        exported_path = _write_json_report(export_json, report_payload)
    if as_json:
        print(json.dumps(report_payload, indent=2))
    else:
        print(format_summary_table(summaries))
        if len(summaries) >= 2:
            improvement = improvement_percent(summaries[0].score_ms, summaries[1].score_ms)
            print(f"improvement_vs_default: {improvement:.1f}% lower weighted latency")
        print(_repro_footer(profile, model, runner))
        if exported_path is not None:
            print(f"exported_json: {exported_path}")
    return 0


def command_ablate(model: Path, as_json: bool, export_json: Path | None) -> int:
    profile = detect_system_profile()
    runner = LlamaCppRunner()
    warning = runner.compatibility_warning()
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    tuned_config, loaded_saved_profile = _resolve_tuned_config(model, profile)
    steps = run_ablation(model, tuned_config, runner=runner)
    stock_score = steps[0].summary.score_ms
    payload_steps = []
    previous_score = stock_score
    for step in steps:
        payload_steps.append(_ablation_step_to_dict(step, stock_score, previous_score))
        previous_score = step.summary.score_ms
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model),
        "used_saved_tuned_profile": loaded_saved_profile,
        "tuned_config": tuned_config.__dict__,
        "plan": [
            {"label": label, "description": description, "overrides": overrides.__dict__}
            for label, description, overrides in ablation_plan(tuned_config)
        ],
        "steps": payload_steps,
        "system": {
            "platform": profile.platform,
            "machine": profile.machine,
            "cpu_brand": profile.cpu_brand,
            "memory_gb": profile.memory_gb,
            "logical_cpu_count": profile.logical_cpu_count,
            "performance_cores": profile.performance_cores,
        },
        "llama_cpp_version": runner.version,
    }
    exported_path: Path | None = None
    if export_json is not None:
        exported_path = _write_json_report(export_json, payload)
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_ablation_table(steps))
        final_improvement = improvement_percent(stock_score, steps[-1].summary.score_ms)
        print(f"final_improvement_vs_stock: {final_improvement:.1f}% lower weighted latency")
        print(_repro_footer(profile, model, runner))
        if exported_path is not None:
            print(f"exported_json: {exported_path}")
    return 0


def command_shapley(model: Path, as_json: bool, export_json: Path | None) -> int:
    profile = detect_system_profile()
    runner = LlamaCppRunner()
    warning = runner.compatibility_warning()
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    tuned_config, loaded_saved_profile = _resolve_tuned_config(model, profile)
    attribution = run_shapley(model, tuned_config, runner=runner)
    total_improvement = attribution.stock_summary.score_ms - attribution.full_summary.score_ms
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model),
        "used_saved_tuned_profile": loaded_saved_profile,
        "tuned_config": tuned_config.__dict__,
        "plan": [
            {"label": label, "description": description, "overrides": overrides.__dict__}
            for label, description, overrides in ablation_plan(tuned_config)
        ],
        "stock_summary": _summary_to_dict(attribution.stock_summary),
        "full_summary": _summary_to_dict(attribution.full_summary),
        "total_improvement_ms": total_improvement,
        "contributions": [_shapley_contribution_to_dict(item) for item in attribution.contributions],
        "subset_summaries": [
            {
                "subset": list(labels),
                "summary": _summary_to_dict(summary),
            }
            for labels, summary in sorted(attribution.subset_summaries.items(), key=lambda item: (len(item[0]), item[0]))
        ],
        "system": {
            "platform": profile.platform,
            "machine": profile.machine,
            "cpu_brand": profile.cpu_brand,
            "memory_gb": profile.memory_gb,
            "logical_cpu_count": profile.logical_cpu_count,
            "performance_cores": profile.performance_cores,
        },
        "llama_cpp_version": runner.version,
    }
    exported_path: Path | None = None
    if export_json is not None:
        exported_path = _write_json_report(export_json, payload)
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_shapley_table(attribution))
        print(f"total_improvement_vs_stock_ms: {total_improvement:.1f}")
        print(_repro_footer(profile, model, runner))
        if exported_path is not None:
            print(f"exported_json: {exported_path}")
    return 0


def command_serve(model: Path, host: str, port: int, print_command: bool) -> int:
    profile = detect_system_profile()
    saved_profile = load_profile(model)
    config = saved_profile.config if saved_profile is not None else heuristic_default_config(profile, model)
    runner = LlamaCppRunner()
    warning = runner.compatibility_warning()
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    command = runner.build_server_command(model, host, port, config)
    if print_command:
        print(" ".join(command))
        return 0
    completed = subprocess.run(command)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return command_inspect(args.json)
        if args.command == "setup":
            return command_setup(args.install_llama_cpp)
        if args.command == "tune":
            return command_tune(args.model, args.max_candidates, args.json)
        if args.command == "benchmark":
            return command_benchmark(args.model, args.include_ollama, args.json, args.export_json)
        if args.command == "ablate":
            return command_ablate(args.model, args.json, args.export_json)
        if args.command == "shapley":
            return command_shapley(args.model, args.json, args.export_json)
        if args.command == "serve":
            return command_serve(args.model, args.host, args.port, args.print_command)
    except BackendError as exc:
        message = str(exc)
        if "llama-cli is not installed or not on PATH" in message:
            message = f"{message}. Run `autoresearch-local setup` for install instructions."
        if "llama-server is not installed or not on PATH" in message:
            message = f"{message}. Run `autoresearch-local setup` for install instructions."
        print(f"error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
