"""Model profile registry: YAML loading and hardware-based auto-selection."""

from __future__ import annotations

import importlib.resources
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from llm_bench.sysinfo import GpuInfo, SystemInfo

if TYPE_CHECKING:
    from llm_bench.devices import RuntimeDevice

_USER_MODELS_DIR = Path.home() / ".llm-bench" / "models"

_ALLOWED_VENDORS = ("amd", "nvidia", "apple", "intel")

# Default backend mapping per vendor when a YAML's gpu_match doesn't list `backends:`.
# Keep aligned with sysinfo.GpuInfo.backend strings ("CUDA", "ROCm", "Metal", "Vulkan").
_VENDOR_DEFAULT_BACKENDS: dict[str, frozenset[str]] = {
    "amd": frozenset({"ROCm", "Vulkan"}),
    "nvidia": frozenset({"CUDA"}),
    "apple": frozenset({"Metal"}),
    "intel": frozenset({"Vulkan"}),
}


@dataclass
class Model:
    name: str
    hf_repo: str = ""
    lm_studio_id: str = ""
    local_path: str = ""
    estimated_size_gb: float = 0.0
    description: str = ""
    tags: list[str] = field(default_factory=list[str])
    extra_args: list[str] = field(default_factory=list[str])

    @property
    def identifier(self) -> str:
        """Stable per-model identifier across backends.

        Priority: LM Studio id → HF repo → local GGUF path. The runner uses
        this string as both the cache key and the display id.
        """
        return self.lm_studio_id or self.hf_repo or self.local_path


@dataclass
class GpuMatch:
    """Criteria for matching a GPU-specific profile to a physical card.

    A profile matches a `GpuInfo` when (a) the vendor's backend mapping (or the
    explicit `backends` list) overlaps with `GpuInfo.backend`, AND (b) at least
    one entry in `name_contains` is a case-insensitive substring of `GpuInfo.name`
    OR `name_regex` matches it, AND (c) `GpuInfo.vram_gb` (when known) meets
    `min_vram_gb`.
    """

    vendor: str  # "amd" | "nvidia" | "apple" | "intel"
    name_contains: list[str] = field(default_factory=list[str])
    name_regex: str = ""
    backends: list[str] = field(default_factory=list[str])
    min_vram_gb: float = 0.0


@dataclass
class ModelProfile:
    profile: str
    description: str
    min_ram_gb: float
    max_ram_gb: float
    models: list[Model]
    source_path: Path | None = None
    gpu_match: GpuMatch | None = None


def _parse_gpu_match(data: dict[str, Any], path: Path) -> GpuMatch:
    vendor = str(data.get("vendor", "")).strip().lower()
    if vendor not in _ALLOWED_VENDORS:
        raise ValueError(
            f"gpu_match.vendor in {path} must be one of {_ALLOWED_VENDORS}, got {vendor!r}"
        )
    name_contains_raw: object = data.get("name_contains", []) or []
    if isinstance(name_contains_raw, str):
        name_contains = [name_contains_raw]
    elif isinstance(name_contains_raw, list):
        name_contains = [str(x) for x in cast(list[Any], name_contains_raw)]
    else:
        raise ValueError(
            f"gpu_match.name_contains in {path} must be a list or string, "
            f"got {type(name_contains_raw).__name__}"
        )
    backends_raw: object = data.get("backends", []) or []
    if isinstance(backends_raw, str):
        backends = [backends_raw]
    elif isinstance(backends_raw, list):
        backends = [str(x) for x in cast(list[Any], backends_raw)]
    else:
        raise ValueError(
            f"gpu_match.backends in {path} must be a list or string, "
            f"got {type(backends_raw).__name__}"
        )
    return GpuMatch(
        vendor=vendor,
        name_contains=name_contains,
        name_regex=str(data.get("name_regex", "")),
        backends=backends,
        min_vram_gb=float(data.get("min_vram_gb", 0)),
    )


def _load_yaml(path: Path) -> ModelProfile:
    with path.open() as fh:
        data = cast(dict[str, Any], yaml.safe_load(fh))
    models: list[Model] = []
    for m in cast(list[dict[str, Any]], data.get("models", [])):
        hf_repo = m.get("hf_repo", "") or ""
        lm_studio_id = m.get("lm_studio_id", "") or ""
        local_path = m.get("local_path", "") or ""
        if not hf_repo and not lm_studio_id and not local_path:
            raise ValueError(
                f"Model entry '{m.get('name', '?')}' in {path} must specify "
                "at least one of hf_repo / lm_studio_id / local_path"
            )
        raw_extras: object = m.get("extra_args", []) or []
        if isinstance(raw_extras, str):
            extra_args = shlex.split(raw_extras)
        elif isinstance(raw_extras, list):
            extra_args = [str(a) for a in cast(list[Any], raw_extras)]
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
                tags=list(m.get("tags", [])),
                extra_args=extra_args,
            )
        )

    gpu_match: GpuMatch | None = None
    raw_match: object = data.get("gpu_match")
    if raw_match is not None:
        if not isinstance(raw_match, dict):
            raise ValueError(
                f"gpu_match in {path} must be a mapping, got {type(raw_match).__name__}"
            )
        gpu_match = _parse_gpu_match(cast(dict[str, Any], raw_match), path)
        # When the file lives under specific/<vendor>/, sanity-check vendor agreement.
        parts = [p.lower() for p in path.parts]
        if "specific" in parts:
            idx = parts.index("specific")
            if idx + 1 < len(parts) and parts[idx + 1] != gpu_match.vendor:
                raise ValueError(
                    f"gpu_match.vendor={gpu_match.vendor!r} in {path} does not match "
                    f"the parent folder 'specific/{parts[idx + 1]}/'"
                )

    return ModelProfile(
        profile=data["profile"],
        description=data["description"],
        min_ram_gb=float(data.get("min_ram_gb", 0)),
        max_ram_gb=float(data.get("max_ram_gb", 9999)),
        models=models,
        source_path=path,
        gpu_match=gpu_match,
    )


def _walk_resource_yamls(traversable: Any) -> list[Path]:
    """Recurse a Traversable, returning real Paths for every .yaml leaf."""
    out: list[Path] = []
    for entry in traversable.iterdir():
        if entry.is_dir():
            out.extend(_walk_resource_yamls(entry))
        elif str(entry).endswith(".yaml"):
            out.append(Path(str(entry)))
    return out


def _bundled_profiles() -> list[ModelProfile]:
    profiles: list[ModelProfile] = []
    try:
        pkg = importlib.resources.files("llm_bench") / "data" / "models"
        for yml in sorted(_walk_resource_yamls(pkg), key=str):
            profiles.append(_load_yaml(yml))
    except Exception:
        # Fallback: resolve relative to this file (editable installs)
        data_dir = Path(__file__).parent / "data" / "models"
        for yml in sorted(data_dir.rglob("*.yaml")):
            profiles.append(_load_yaml(yml))
    return profiles


def _user_profiles() -> list[ModelProfile]:
    if not _USER_MODELS_DIR.exists():
        return []
    return [_load_yaml(p) for p in sorted(_USER_MODELS_DIR.rglob("*.yaml"))]


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

    Backend-specific profiles (e.g. `lm_studio`) and GPU-specific profiles
    (those with a `gpu_match` block) are excluded — the former are picked
    explicitly via `--backend`, the latter by `matching_gpu_profiles()`.
    """
    profiles = [p for p in all_profiles() if p.profile != "lm_studio" and p.gpu_match is None]
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


def gpu_matches_profile(match: GpuMatch, gpu: GpuInfo) -> bool:
    """True iff the given physical GPU satisfies `match`."""
    allowed_backends = (
        {b for b in match.backends}
        if match.backends
        else set(_VENDOR_DEFAULT_BACKENDS.get(match.vendor, frozenset()))
    )
    if allowed_backends and gpu.backend not in allowed_backends:
        return False
    name_lower = gpu.name.lower()
    matched_name = False
    if match.name_contains:
        for needle in match.name_contains:
            if needle.lower() in name_lower:
                matched_name = True
                break
    if match.name_regex:
        if re.search(match.name_regex, gpu.name, flags=re.IGNORECASE):
            matched_name = True
    if not match.name_contains and not match.name_regex:
        matched_name = True  # vendor-only match
    if not matched_name:
        return False
    if match.min_vram_gb > 0 and gpu.vram_gb is not None:
        if gpu.vram_gb + 0.5 < match.min_vram_gb:  # 0.5 GB rounding tolerance
            return False
    return True


def _runtime_devices_satisfy(
    match: GpuMatch,
    runtime_devices: list[RuntimeDevice] | None,
) -> bool:
    """True iff the runtime probe sees a device compatible with this profile.

    `runtime_devices is None` means "couldn't query the active backend" — in
    that case we warn-and-proceed (return True) so a missing `--list-devices`
    flag on an old binary doesn't silently disable GPU profiles.
    """
    if runtime_devices is None:
        return True
    if not runtime_devices:
        return False
    allowed_backends = (
        {b for b in match.backends}
        if match.backends
        else set(_VENDOR_DEFAULT_BACKENDS.get(match.vendor, frozenset()))
    )
    for dev in runtime_devices:
        if allowed_backends and dev.backend not in allowed_backends:
            continue
        if not match.name_contains and not match.name_regex:
            return True
        desc_lower = dev.description.lower()
        if any(n.lower() in desc_lower for n in match.name_contains):
            return True
        if match.name_regex and re.search(match.name_regex, dev.description, flags=re.IGNORECASE):
            return True
    return False


def find_matched_runtime_device(
    match: GpuMatch,
    runtime_devices: list[RuntimeDevice] | None,
) -> RuntimeDevice | None:
    """Return the specific RuntimeDevice satisfying `match`, or None.

    Used by callers that need the device's index to set `HIP_VISIBLE_DEVICES`
    / `CUDA_VISIBLE_DEVICES` (see `devices.auto_env_for_device`). Returns
    None when `runtime_devices` is None (probe couldn't run) so callers know
    they can't auto-pin.
    """
    if runtime_devices is None:
        return None
    allowed_backends = (
        {b for b in match.backends}
        if match.backends
        else set(_VENDOR_DEFAULT_BACKENDS.get(match.vendor, frozenset()))
    )
    for dev in runtime_devices:
        if allowed_backends and dev.backend not in allowed_backends:
            continue
        if not match.name_contains and not match.name_regex:
            return dev  # vendor-only match
        desc_lower = dev.description.lower()
        if any(n.lower() in desc_lower for n in match.name_contains):
            return dev
        if match.name_regex and re.search(match.name_regex, dev.description, flags=re.IGNORECASE):
            return dev
    return None


def matching_gpu_profiles(
    sysinfo: SystemInfo,
    runtime_devices: list[RuntimeDevice] | None,
) -> list[ModelProfile]:
    """All GPU-specific profiles satisfied by both physical GPUs and the runtime probe.

    `runtime_devices` is the result of probing the active backend (e.g.
    `devices.llama_bench_devices()`). `None` means the probe couldn't run
    (e.g. older binary that rejects `--list-devices`); in that case the
    matcher warns-and-proceeds with physical-detection only.

    The returned list is sorted by `min_vram_gb` of the gpu_match block
    (descending) so the most VRAM-restrictive — i.e. most specific — profile
    comes first. Callers using "replace" semantics should pick the head.
    """
    out: list[ModelProfile] = []
    for p in all_profiles():
        if p.gpu_match is None:
            continue
        physical_match = any(gpu_matches_profile(p.gpu_match, g) for g in sysinfo.gpus)
        if not physical_match:
            continue
        if not _runtime_devices_satisfy(p.gpu_match, runtime_devices):
            continue
        out.append(p)
    out.sort(key=lambda p: -(p.gpu_match.min_vram_gb if p.gpu_match else 0.0))
    return out
