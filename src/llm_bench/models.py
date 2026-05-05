"""Model profile registry: YAML loading and hardware-based auto-selection."""

from __future__ import annotations

import importlib.resources
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from llm_bench.sysinfo import SystemInfo

_USER_MODELS_DIR = Path.home() / ".llm-bench" / "models"


@dataclass
class Model:
    name: str
    hf_repo: str = ""
    lm_studio_id: str = ""
    local_path: str = ""
    estimated_size_gb: float = 0.0
    description: str = ""
    tags: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)

    @property
    def identifier(self) -> str:
        """Stable per-model identifier across backends.

        Priority: LM Studio id → HF repo → local GGUF path. The runner uses
        this string as both the cache key and the display id.
        """
        return self.lm_studio_id or self.hf_repo or self.local_path


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
    models: list[Model] = []
    for m in data.get("models", []):
        hf_repo = m.get("hf_repo", "") or ""
        lm_studio_id = m.get("lm_studio_id", "") or ""
        local_path = m.get("local_path", "") or ""
        if not hf_repo and not lm_studio_id and not local_path:
            raise ValueError(
                f"Model entry '{m.get('name', '?')}' in {path} must specify "
                "at least one of hf_repo / lm_studio_id / local_path"
            )
        raw_extras = m.get("extra_args", []) or []
        if isinstance(raw_extras, str):
            extra_args = shlex.split(raw_extras)
        elif isinstance(raw_extras, list):
            extra_args = [str(a) for a in raw_extras]
        else:
            raise ValueError(
                f"Model entry '{m.get('name', '?')}' in {path}: extra_args must be a "
                f"list or string, got {type(raw_extras).__name__}"
            )
        models.append(
            Model(
                name=m["name"],
                hf_repo=hf_repo,
                lm_studio_id=lm_studio_id,
                local_path=local_path,
                estimated_size_gb=float(m.get("estimated_size_gb", 0)),
                description=m.get("description", ""),
                tags=m.get("tags", []),
                extra_args=extra_args,
            )
        )
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
    """Pick the highest-tier profile whose min_ram_gb fits total system RAM.

    Backend-specific profiles (e.g. `lm_studio`) are excluded from RAM-based
    auto-selection — they're picked explicitly via `--backend` instead.
    """
    profiles = [p for p in all_profiles() if p.profile != "lm_studio"]
    ram = sysinfo.total_ram_gb
    for p in reversed(profiles):
        if p.min_ram_gb <= ram:
            return p
    return profiles[0]


def load_profile_from_file(path: Path) -> ModelProfile:
    return _load_yaml(path)


def get_profile_by_name(name: str) -> ModelProfile | None:
    """Return the first profile (user override, then bundled) matching `name`, or None."""
    for p in all_profiles():
        if p.profile == name:
            return p
    return None
