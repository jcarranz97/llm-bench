"""Model profile registry: YAML loading and hardware-based auto-selection."""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from llm_bench.sysinfo import SystemInfo

_USER_MODELS_DIR = Path.home() / ".llm-bench" / "models"


@dataclass
class Model:
    name: str
    hf_repo: str
    estimated_size_gb: float
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class ModelProfile:
    profile: str
    description: str
    min_ram_gb: float
    max_ram_gb: float
    models: list[Model]
    source_path: Path | None = None


def _load_yaml(path: Path) -> ModelProfile:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    models = [
        Model(
            name=m["name"],
            hf_repo=m["hf_repo"],
            estimated_size_gb=float(m.get("estimated_size_gb", 0)),
            description=m.get("description", ""),
            tags=m.get("tags", []),
        )
        for m in data.get("models", [])
    ]
    return ModelProfile(
        profile=data["profile"],
        description=data["description"],
        min_ram_gb=float(data.get("min_ram_gb", 0)),
        max_ram_gb=float(data.get("max_ram_gb", 9999)),
        models=models,
        source_path=path,
    )


def _bundled_profiles() -> list[ModelProfile]:
    profiles: list[ModelProfile] = []
    try:
        pkg = importlib.resources.files("llm_bench") / "data" / "models"
        for resource in pkg.iterdir():  # type: ignore[union-attr]
            if str(resource).endswith(".yaml"):
                profiles.append(_load_yaml(Path(str(resource))))
    except Exception:
        # Fallback: resolve relative to this file (editable installs)
        data_dir = Path(__file__).parent / "data" / "models"
        for yml in sorted(data_dir.glob("*.yaml")):
            profiles.append(_load_yaml(yml))
    return profiles


def _user_profiles() -> list[ModelProfile]:
    if not _USER_MODELS_DIR.exists():
        return []
    return [_load_yaml(p) for p in sorted(_USER_MODELS_DIR.glob("*.yaml"))]


def all_profiles() -> list[ModelProfile]:
    """Return user profiles (override) + bundled profiles, sorted by min_ram_gb."""
    seen: set[str] = set()
    result: list[ModelProfile] = []
    for p in _user_profiles() + _bundled_profiles():
        if p.profile not in seen:
            seen.add(p.profile)
            result.append(p)
    return sorted(result, key=lambda p: p.min_ram_gb)


def select_profile(sysinfo: SystemInfo) -> ModelProfile:
    """Pick the highest-tier profile whose max_ram_gb fits total system RAM."""
    profiles = all_profiles()
    ram = sysinfo.total_ram_gb
    # Walk from highest to lowest and take the first one that fits
    for p in reversed(profiles):
        if p.min_ram_gb <= ram:
            return p
    return profiles[0]


def load_profile_from_file(path: Path) -> ModelProfile:
    return _load_yaml(path)
