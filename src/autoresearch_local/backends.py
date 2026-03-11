from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .profiles import LlamaCppConfig, LlamaCppOverrides


TESTED_LLAMA_CPP_BUILD = "b8260-96cfc4992"
PROMPT_TPS_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second",
    re.IGNORECASE | re.DOTALL,
)
DECODE_TPS_RE = re.compile(
    r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second",
    re.IGNORECASE | re.DOTALL,
)
TOTAL_TIME_RE = re.compile(r"total time\s*=\s*([\d.]+)\s*ms", re.IGNORECASE)
COMPACT_TPS_RE = re.compile(
    r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]",
    re.IGNORECASE,
)
CORRUPT_MODEL_PATTERNS = (
    "model is corrupted or incomplete",
    "data is not within the file bounds",
)
NOISE_LINE_PREFIXES = (
    "ggml_metal_",
    "ggml_backend_",
    "ggml_cuda_",
    "llama_memory_breakdown_print:",
)


class BackendError(RuntimeError):
    """Raised when a backend command fails or cannot be parsed."""


@dataclass(frozen=True)
class BenchmarkRun:
    total_ms: float
    prompt_tps: float
    decode_tps: float
    prompt_tokens: int
    decode_tokens: int
    raw_output: str


def _humanize_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def summarize_model_load_failure(model_path: Path, output: str) -> str:
    lowered = output.lower()
    if not any(pattern in lowered for pattern in CORRUPT_MODEL_PATTERNS):
        return output.strip() or "llama.cpp benchmark failed"
    relevant_lines = []
    for line in output.splitlines():
        stripped = line.strip()
        lowered_line = stripped.lower()
        if any(pattern in lowered_line for pattern in CORRUPT_MODEL_PATTERNS):
            relevant_lines.append(stripped)
        elif "failed to load model" in lowered_line and stripped not in relevant_lines:
            relevant_lines.append(stripped)
    detail = "\n".join(f"- {line}" for line in relevant_lines[:3])
    try:
        size_text = _humanize_bytes(model_path.stat().st_size)
    except OSError:
        size_text = "unknown size"
    return (
        f"GGUF load failed for {model_path} ({size_text}). "
        "The file appears incomplete or corrupted. Re-download the model and retry.\n"
        "Note: the `tensor API disabled for pre-M5 and pre-A19 devices` Metal line is informational on M-series Macs and is not the root cause.\n"
        f"{detail}"
    )


def summarize_llama_cpp_failure(model_path: Path, output: str) -> str:
    lowered = output.lower()
    if any(pattern in lowered for pattern in CORRUPT_MODEL_PATTERNS):
        return summarize_model_load_failure(model_path, output)
    relevant_lines: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered_line = stripped.lower()
        if lowered_line.startswith(NOISE_LINE_PREFIXES):
            continue
        if stripped in {"usage:", "to show complete usage, run with -h"}:
            continue
        if any(
            token in lowered_line
            for token in (
                "error while handling argument",
                "failed to load model",
                "error:",
                "invalid value",
                "expected value for argument",
                "not supported",
            )
        ):
            relevant_lines.append(stripped)
    if relevant_lines:
        return "\n".join(relevant_lines[:3])
    return output.strip() or "llama.cpp benchmark failed"


class LlamaCppRunner:
    def __init__(self) -> None:
        self.version = self._get_version("llama-cli")
        self.help_text = self._get_help_text("llama-cli")
        self.server_help_text: str | None = None

    @staticmethod
    def _get_version(command: str) -> str | None:
        try:
            completed = subprocess.run(
                [command, "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("version:"):
                return stripped
        return output.splitlines()[0] if output else None

    @staticmethod
    def _get_help_text(command: str) -> str:
        try:
            completed = subprocess.run(
                [command, "--help"],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise BackendError(f"{command} is not installed or not on PATH") from exc
        except subprocess.CalledProcessError as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return output
        return (completed.stdout or "") + (completed.stderr or "")

    @staticmethod
    def _resolve_flag(help_text: str, *aliases: str) -> str | None:
        for alias in aliases:
            if alias in help_text:
                return alias
        return None

    def _build_common_args(self, config: LlamaCppConfig | LlamaCppOverrides | None, help_text: str) -> list[str]:
        args: list[str] = []
        if config is None:
            return args
        thread_flag = self._resolve_flag(help_text, "--threads", "-t")
        ctx_flag = self._resolve_flag(help_text, "--ctx-size", "-c")
        batch_flag = self._resolve_flag(help_text, "--batch-size", "-b")
        ubatch_flag = self._resolve_flag(help_text, "--ubatch-size", "-ub", "--ubatch-size")
        gpu_layers_flag = self._resolve_flag(help_text, "--n-gpu-layers", "-ngl")
        flash_flag = self._resolve_flag(help_text, "--flash-attn", "-fa")
        if thread_flag and getattr(config, "threads", None) is not None:
            args.extend([thread_flag, str(config.threads)])
        if ctx_flag and getattr(config, "ctx_size", None) is not None:
            args.extend([ctx_flag, str(config.ctx_size)])
        if batch_flag and getattr(config, "batch_size", None) is not None:
            args.extend([batch_flag, str(config.batch_size)])
        if ubatch_flag and getattr(config, "ubatch_size", None) is not None:
            args.extend([ubatch_flag, str(config.ubatch_size)])
        if gpu_layers_flag and getattr(config, "gpu_layers", None) is not None:
            args.extend([gpu_layers_flag, str(config.gpu_layers)])
        if flash_flag and getattr(config, "flash_attention", None) is not None:
            args.extend([flash_flag, "on" if config.flash_attention else "off"])
        return args

    def build_cli_command(
        self,
        model_path: Path,
        prompt: str,
        max_tokens: int,
        config: LlamaCppConfig | LlamaCppOverrides | None,
    ) -> list[str]:
        prompt_flag = self._resolve_flag(self.help_text, "--prompt", "-p")
        max_tokens_flag = self._resolve_flag(self.help_text, "--n-predict", "-n")
        seed_flag = self._resolve_flag(self.help_text, "--seed", "-s")
        temp_flag = self._resolve_flag(self.help_text, "--temp")
        top_k_flag = self._resolve_flag(self.help_text, "--top-k")
        top_p_flag = self._resolve_flag(self.help_text, "--top-p")
        model_flag = self._resolve_flag(self.help_text, "--model", "-m")
        no_display_flag = self._resolve_flag(self.help_text, "--no-display-prompt")
        single_turn_flag = self._resolve_flag(self.help_text, "--single-turn", "-st")
        simple_io_flag = self._resolve_flag(self.help_text, "--simple-io")
        no_warmup_flag = self._resolve_flag(self.help_text, "--no-warmup")
        args = ["llama-cli"]
        if model_flag:
            args.extend([model_flag, str(model_path)])
        if prompt_flag:
            args.extend([prompt_flag, prompt])
        if max_tokens_flag:
            args.extend([max_tokens_flag, str(max_tokens)])
        if seed_flag:
            args.extend([seed_flag, "42"])
        if temp_flag:
            args.extend([temp_flag, "0"])
        if top_k_flag:
            args.extend([top_k_flag, "1"])
        if top_p_flag:
            args.extend([top_p_flag, "1"])
        if no_display_flag:
            args.append(no_display_flag)
        if single_turn_flag:
            args.append(single_turn_flag)
        if simple_io_flag:
            args.append(simple_io_flag)
        if no_warmup_flag:
            args.append(no_warmup_flag)
        args.extend(self._build_common_args(config, self.help_text))
        return args

    def build_server_command(
        self,
        model_path: Path,
        host: str,
        port: int,
        config: LlamaCppConfig | LlamaCppOverrides | None,
    ) -> list[str]:
        if self.server_help_text is None:
            self.server_help_text = self._get_help_text("llama-server")
        model_flag = self._resolve_flag(self.server_help_text, "--model", "-m")
        host_flag = self._resolve_flag(self.server_help_text, "--host")
        port_flag = self._resolve_flag(self.server_help_text, "--port")
        args = ["llama-server"]
        if model_flag:
            args.extend([model_flag, str(model_path)])
        if host_flag:
            args.extend([host_flag, host])
        if port_flag:
            args.extend([port_flag, str(port)])
        args.extend(self._build_common_args(config, self.server_help_text))
        return args

    @staticmethod
    def parse_cli_output(output: str, elapsed_ms: float) -> BenchmarkRun:
        prompt_match = PROMPT_TPS_RE.search(output)
        decode_match = DECODE_TPS_RE.search(output)
        total_match = TOTAL_TIME_RE.search(output)
        compact_match = COMPACT_TPS_RE.search(output)
        if prompt_match and decode_match:
            total_ms = float(total_match.group(1)) if total_match else elapsed_ms
            return BenchmarkRun(
                total_ms=total_ms,
                prompt_tps=float(prompt_match.group(3)),
                decode_tps=float(decode_match.group(3)),
                prompt_tokens=int(prompt_match.group(2)),
                decode_tokens=int(decode_match.group(2)),
                raw_output=output,
            )
        if compact_match:
            return BenchmarkRun(
                total_ms=elapsed_ms,
                prompt_tps=float(compact_match.group(1)),
                decode_tps=float(compact_match.group(2)),
                prompt_tokens=0,
                decode_tokens=0,
                raw_output=output,
            )
        raise BackendError("Unable to parse llama.cpp timing output")

    def run_once(
        self,
        model_path: Path,
        prompt: str,
        max_tokens: int,
        config: LlamaCppConfig | LlamaCppOverrides | None,
        timeout_seconds: int,
    ) -> BenchmarkRun:
        command = self.build_cli_command(model_path, prompt, max_tokens, config)
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"llama.cpp benchmark timed out after {timeout_seconds}s") from exc
        except subprocess.CalledProcessError as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            raise BackendError(summarize_llama_cpp_failure(model_path, output)) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000
        output = (completed.stdout or "") + (completed.stderr or "")
        return self.parse_cli_output(output, elapsed_ms)

    def compatibility_warning(self) -> str | None:
        if self.version and not _matches_tested_build(self.version, TESTED_LLAMA_CPP_BUILD):
            return (
                f"Detected llama.cpp `{self.version}`. "
                f"`autoresearch-local` is currently tested against `{TESTED_LLAMA_CPP_BUILD}`; "
                "other builds may work, but CLI flags and timing output can differ."
            )
        return None


def _matches_tested_build(version: str, tested_build: str) -> bool:
    version_match = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", version, re.IGNORECASE)
    tested_match = re.search(r"b(\d+)-([0-9a-f]+)", tested_build, re.IGNORECASE)
    if version_match and tested_match:
        return version_match.group(1) == tested_match.group(1) and version_match.group(2) == tested_match.group(2)
    return tested_build in version


class OllamaRunner:
    def __init__(self, host: str = "http://127.0.0.1:11434") -> None:
        self.host = host.rstrip("/")
        if not shutil.which("ollama"):
            raise BackendError("ollama is not installed or not on PATH")

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise BackendError(
                "unable to reach Ollama API at http://127.0.0.1:11434; make sure Ollama is running"
            ) from exc

    def ensure_alive(self) -> None:
        self._request("GET", "/api/tags")

    def ensure_model(self, model_path: Path, model_name: str) -> str:
        self.ensure_alive()
        models = self._request("GET", "/api/tags").get("models", [])
        if any(model.get("name") == model_name for model in models):
            return model_name
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(f"FROM {model_path.resolve()}\n")
            modelfile_path = Path(handle.name)
        completed = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        modelfile_path.unlink(missing_ok=True)
        if completed.returncode != 0:
            raise BackendError("failed to create Ollama model from GGUF")
        return model_name

    @staticmethod
    def parse_generate_response(payload: dict) -> BenchmarkRun:
        total_ms = payload["total_duration"] / 1_000_000
        prompt_tps = 0.0
        decode_tps = 0.0
        prompt_count = int(payload.get("prompt_eval_count", 0))
        decode_count = int(payload.get("eval_count", 0))
        prompt_duration = payload.get("prompt_eval_duration", 0)
        decode_duration = payload.get("eval_duration", 0)
        if prompt_duration:
            prompt_tps = prompt_count / (prompt_duration / 1_000_000_000)
        if decode_duration:
            decode_tps = decode_count / (decode_duration / 1_000_000_000)
        return BenchmarkRun(
            total_ms=total_ms,
            prompt_tps=prompt_tps,
            decode_tps=decode_tps,
            prompt_tokens=prompt_count,
            decode_tokens=decode_count,
            raw_output=json.dumps(payload),
        )

    def run_once(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int,
        config: LlamaCppConfig | None,
    ) -> BenchmarkRun:
        options = {
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "num_predict": max_tokens,
        }
        if config is not None:
            options.update(config.to_ollama_options())
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        response = self._request("POST", "/api/generate", payload)
        return self.parse_generate_response(response)
