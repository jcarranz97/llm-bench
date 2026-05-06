"""Probe the active backend for the GPU device list it actually sees.

Each function returns a `list[RuntimeDevice]` on success, or `None` when the
probe couldn't run (older binary that rejects the flag, missing endpoint,
absent CLI tool). `None` is treated as "couldn't verify, warn and proceed"
by `models.matching_gpu_profiles()` — that way an old llama-bench build or a
machine without `lms` installed doesn't silently disable GPU profiles.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from llm_bench.runner import build_env

if TYPE_CHECKING:
    from llm_bench.llama_server import LlamaServerClient
    from llm_bench.sysinfo import SystemInfo


@dataclass
class RuntimeDevice:
    """One device the active backend reports as available for inference."""

    name: str  # "ROCm0", "CUDA0", "Metal", "CPU", ...
    description: str  # "AMD Radeon RX 7900 XT"
    backend: str  # ROCm | CUDA | Metal | Vulkan | CPU


# Map a llama-bench / lms runtime prefix to a llm_bench backend label.
_PREFIX_TO_BACKEND = {
    "rocm": "ROCm",
    "cuda": "CUDA",
    "vulkan": "Vulkan",
    "metal": "Metal",
    "sycl": "Vulkan",  # closest analogue for our matching map
    "cpu": "CPU",
}


def _prefix_backend(name: str) -> str:
    """ROCm0 → ROCm, CUDA1 → CUDA, vulkan-llama.cpp → Vulkan, ..."""
    lower = name.lower()
    for prefix, backend in _PREFIX_TO_BACKEND.items():
        if lower.startswith(prefix):
            return backend
    return "CPU"


# Vendor → env-var name used by llama.cpp / GGML to pin to one device.
_VENDOR_VISIBLE_ENV = {
    "amd": "HIP_VISIBLE_DEVICES",
    "nvidia": "CUDA_VISIBLE_DEVICES",
    "intel": "GGML_VK_VISIBLE_DEVICES",
}


def device_index(device: RuntimeDevice) -> int | None:
    """Extract the trailing integer index from a device name like 'ROCm0' → 0.

    Returns None when the name has no trailing digits (e.g. 'CPU', 'Metal').
    """
    m = re.search(r"(\d+)\s*$", device.name)
    return int(m.group(1)) if m else None


def auto_env_for_device(device: RuntimeDevice, vendor: str) -> dict[str, str]:
    """Vendor-aware env var to pin llama.cpp to the matched device.

    AMD ROCm/Vulkan → `HIP_VISIBLE_DEVICES=N`
    NVIDIA CUDA     → `CUDA_VISIBLE_DEVICES=N`
    Intel Vulkan    → `GGML_VK_VISIBLE_DEVICES=N`

    Returns an empty dict when the device has no extractable index (e.g. CPU,
    Metal — Metal exposes a single device, no env-pinning needed) or when the
    vendor doesn't have a canonical env var (Apple).
    """
    idx = device_index(device)
    if idx is None:
        return {}
    env_name = _VENDOR_VISIBLE_ENV.get(vendor)
    if not env_name:
        return {}
    return {env_name: str(idx)}


def llama_bench_devices(
    llama_bench: str,
    env_vars: Mapping[str, str] | None = None,
) -> list[RuntimeDevice] | None:
    """Run `<llama-bench> --list-devices`, parse the human-readable output.

    Returns `None` when the binary rejects the flag (older builds) or when the
    output is unparseable, so the matcher can fall back to physical-only.
    """
    try:
        result = subprocess.run(
            [llama_bench, "--list-devices"],
            capture_output=True,
            text=True,
            timeout=10,
            env=build_env(env_vars),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or result.stderr).strip()
    if not out:
        return None

    # Output shape varies across llama.cpp versions, but the canonical form is:
    #     Available devices:
    #       ROCm0: AMD Radeon RX 7900 XT (20464 MiB, 20256 MiB free)
    #       CPU: AMD Ryzen 9 7950X
    line_re = re.compile(r"^\s*(?P<name>[A-Za-z]+\d*|CPU)\s*:\s*(?P<desc>.+?)\s*(?:\(.*\))?\s*$")
    devices: list[RuntimeDevice] = []
    for raw in out.splitlines():
        m = line_re.match(raw)
        if not m:
            continue
        name = m.group("name")
        desc = m.group("desc").strip()
        if not desc or name.lower() == "available":  # skip the header sentinel
            continue
        devices.append(RuntimeDevice(name=name, description=desc, backend=_prefix_backend(name)))
    return devices or None


def llama_server_devices(client: LlamaServerClient) -> list[RuntimeDevice] | None:
    """Read `/props.devices` if the server exposes it (newer llama-server builds)."""
    from llm_bench.llama_server import LlamaServerError

    try:
        props = client.props()
    except LlamaServerError:
        return None
    raw = props.get("devices")
    if not isinstance(raw, list) or not raw:
        return None
    devices: list[RuntimeDevice] = []
    for entry_raw in cast(list[Any], raw):
        if not isinstance(entry_raw, dict):
            continue
        entry = cast(dict[str, Any], entry_raw)
        name = str(entry.get("name") or entry.get("id") or "")
        desc = str(entry.get("description") or entry.get("name") or name)
        backend = str(entry.get("backend") or _prefix_backend(name))
        if not name and not desc:
            continue
        devices.append(RuntimeDevice(name=name or desc, description=desc, backend=backend))
    return devices or None


def _lms_runtime_ids() -> list[str] | None:
    """Run `lms runtime ls`, return the runtime IDs as a list (or None on failure)."""
    try:
        result = subprocess.run(
            ["lms", "runtime", "ls"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or result.stderr).strip()
    if not out:
        return None

    # `lms runtime ls` prints one entry per line; older versions may include
    # a leading header. Capture tokens that look like runtime IDs (contain
    # "llama.cpp" or a known backend prefix).
    ids: list[str] = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith(("name", "runtime", "id", "─", "-")):
            continue
        token = line.split()[0]
        if "llama.cpp" in token.lower() or any(
            token.lower().startswith(p) for p in _PREFIX_TO_BACKEND
        ):
            ids.append(token)
    return ids or None


def lm_studio_runtimes(sysinfo: SystemInfo) -> list[RuntimeDevice] | None:
    """Synthesise a device list from `lms runtime ls` ∩ physical GPUs.

    LM Studio doesn't expose a per-card device list, so we use the installed
    runtimes as the proxy: `rocm-llama.cpp` installed → AMD GPUs are usable;
    `cuda-llama.cpp` installed → NVIDIA usable; etc. The returned list pairs
    each compatible runtime with each matching physical GPU so the matcher
    in `models.py` can cross-check vendor and name.
    """
    runtime_ids = _lms_runtime_ids()
    if runtime_ids is None:
        return None
    runtime_backends: set[str] = set()
    for rid in runtime_ids:
        backend = _prefix_backend(rid)
        if backend != "CPU":
            runtime_backends.add(backend)
    if not runtime_backends:
        return []
    devices: list[RuntimeDevice] = []
    for gpu in sysinfo.gpus:
        if gpu.backend in runtime_backends:
            devices.append(
                RuntimeDevice(
                    name=f"{gpu.backend}{len(devices)}",
                    description=gpu.name,
                    backend=gpu.backend,
                )
            )
    return devices or None
