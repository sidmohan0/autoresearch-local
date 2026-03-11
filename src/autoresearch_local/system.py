from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    executable: str | None
    version: str | None = None


@dataclass(frozen=True)
class DiscoveredModel:
    path: str
    size_bytes: int


@dataclass(frozen=True)
class SystemProfile:
    platform: str
    machine: str
    processor: str
    cpu_brand: str
    memory_gb: float
    logical_cpu_count: int
    performance_cores: int

    @property
    def is_apple_silicon(self) -> bool:
        return self.platform == "Darwin" and self.machine == "arm64"


def _run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def detect_command_version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.SubprocessError:
        return None
    output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    for line in output.splitlines():
        stripped = line.strip()
        if re.match(r"^(version:|ollama version is)", stripped, re.IGNORECASE):
            return stripped
    first_line = output.splitlines()[0].strip() if output else ""
    return first_line or None


def detect_system_profile() -> SystemProfile:
    cpu_brand = ""
    memory_bytes = 0
    performance_cores = 0
    if platform.system() == "Darwin":
        try:
            cpu_brand = _run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
        except subprocess.SubprocessError:
            cpu_brand = platform.processor()
        try:
            memory_bytes = int(_run_text(["sysctl", "-n", "hw.memsize"]))
        except subprocess.SubprocessError:
            memory_bytes = 0
        try:
            performance_cores = int(_run_text(["sysctl", "-n", "hw.perflevel0.physicalcpu"]))
        except subprocess.SubprocessError:
            performance_cores = 0
    return SystemProfile(
        platform=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_brand=cpu_brand or platform.processor(),
        memory_gb=round(memory_bytes / (1024**3), 1) if memory_bytes else 0.0,
        logical_cpu_count=os.cpu_count() or 1,
        performance_cores=performance_cores or max(1, (os.cpu_count() or 2) // 2),
    )


def detect_backends() -> dict[str, BackendStatus]:
    brew_path = shutil.which("brew")
    llama_cli_path = shutil.which("llama-cli")
    llama_server_path = shutil.which("llama-server")
    ollama_path = shutil.which("ollama")
    return {
        "brew": BackendStatus("brew", brew_path is not None, brew_path),
        "llama-cli": BackendStatus(
            "llama-cli",
            llama_cli_path is not None,
            llama_cli_path,
            detect_command_version(["llama-cli", "--version"]) if llama_cli_path else None,
        ),
        "llama-server": BackendStatus(
            "llama-server",
            llama_server_path is not None,
            llama_server_path,
            detect_command_version(["llama-server", "--version"]) if llama_server_path else None,
        ),
        "ollama": BackendStatus(
            "ollama",
            ollama_path is not None,
            ollama_path,
            detect_command_version(["ollama", "--version"]) if ollama_path else None,
        ),
    }


def _env_model_dirs() -> list[Path]:
    raw = os.environ.get("AUTORESEARCH_LOCAL_MODEL_DIRS", "").strip()
    if not raw:
        return []
    return [Path(item).expanduser().resolve() for item in raw.split(os.pathsep) if item.strip()]


def candidate_model_roots(cwd: Path | None = None) -> list[Path]:
    current = (cwd or Path.cwd()).resolve()
    roots = [
        current,
        current.parent,
        Path.home() / "models",
        Path.home() / "Downloads",
        Path.home() / "Documents",
        Path.home() / "Desktop",
        Path.home() / "projects",
    ]
    roots.extend(_env_model_dirs())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.exists() or not root.is_dir():
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def discover_models(
    cwd: Path | None = None,
    max_depth: int = 3,
    max_results: int = 50,
) -> list[DiscoveredModel]:
    discovered: list[DiscoveredModel] = []
    for root in candidate_model_roots(cwd):
        for directory, dirnames, filenames in os.walk(root):
            current_dir = Path(directory)
            depth = len(current_dir.relative_to(root).parts)
            if depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            for filename in sorted(filenames):
                if not filename.lower().endswith(".gguf"):
                    continue
                path = current_dir / filename
                try:
                    size_bytes = path.stat().st_size
                except OSError:
                    continue
                discovered.append(DiscoveredModel(path=str(path), size_bytes=size_bytes))
                if len(discovered) >= max_results:
                    return sorted(discovered, key=lambda item: item.path)
    return sorted(discovered, key=lambda item: item.path)


def format_inspect_text(
    profile: SystemProfile,
    backends: dict[str, BackendStatus],
    models: list[DiscoveredModel],
) -> str:
    backend_lines = []
    for name in sorted(backends):
        status = backends[name]
        location = status.executable or "missing"
        version = f", version={status.version}" if status.version else ""
        backend_lines.append(f"{name}: {'available' if status.available else 'missing'} ({location}{version})")
    model_lines = [f"models_found: {len(models)}"]
    model_lines.extend([f"model: {model.path} ({model.size_bytes} bytes)" for model in models])
    return "\n".join(
        [
            f"platform: {profile.platform}",
            f"machine: {profile.machine}",
            f"cpu: {profile.cpu_brand or profile.processor}",
            f"memory_gb: {profile.memory_gb}",
            f"logical_cpu_count: {profile.logical_cpu_count}",
            f"performance_cores: {profile.performance_cores}",
            *backend_lines,
            *model_lines,
        ]
    )


def format_inspect_json(
    profile: SystemProfile,
    backends: dict[str, BackendStatus],
    models: list[DiscoveredModel],
) -> str:
    payload = {
        "system": asdict(profile),
        "backends": {name: asdict(status) for name, status in backends.items()},
        "models": [asdict(model) for model in models],
    }
    return json.dumps(payload, indent=2)
