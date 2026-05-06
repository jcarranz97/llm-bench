"""Hardware detection: CPU, RAM, GPU."""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


@dataclass
class GpuInfo:
    name: str
    vram_gb: float | None = None
    backend: str = "CPU"  # CUDA | ROCm | Metal | CPU


@dataclass
class SystemInfo:
    cpu_model: str
    cpu_cores: int  # logical threads
    total_ram_gb: float
    available_ram_gb: float
    os: str
    gpus: list[GpuInfo] = field(default_factory=list[GpuInfo])
    cpu_physical_cores: int = 0  # physical cores (0 = unknown)

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)

    @property
    def fingerprint(self) -> str:
        ram_rounded = round(self.total_ram_gb)
        gpu_str = "|".join(f"{g.name}:{g.vram_gb}" for g in self.gpus) or "none"
        raw = f"{self.cpu_model}:{self.cpu_cores}:{ram_rounded}:{gpu_str}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _proc_cpuinfo_model() -> str:
    """Read CPU model name from /proc/cpuinfo (Linux/WSL)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _detect_cpu_model() -> str:
    try:
        import cpuinfo  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

        info = cast(dict[str, Any], cpuinfo.get_cpu_info())  # pyright: ignore[reportUnknownMemberType]
        brand = str(info.get("brand_raw", ""))
        # py-cpuinfo sometimes returns the arch string (e.g. "x86_64") on WSL;
        # treat that as a miss and fall through to /proc/cpuinfo.
        if brand and brand not in ("x86_64", "aarch64", "arm64"):
            return brand
    except ImportError:
        pass

    if name := _proc_cpuinfo_model():
        return name

    return platform.processor() or "Unknown CPU"


def _detect_nvidia_gpus() -> list[GpuInfo]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        gpus: list[GpuInfo] = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            name = parts[0] if parts else "Unknown NVIDIA GPU"
            vram = float(parts[1]) / 1024 if len(parts) > 1 else None
            gpus.append(GpuInfo(name=name, vram_gb=vram, backend="CUDA"))
        return gpus
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def _detect_amd_gpus() -> list[GpuInfo]:
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--csv"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        gpus: list[GpuInfo] = []
        for line in out.strip().splitlines()[1:]:  # skip header
            name = line.strip().split(",")[-1].strip()
            if name:
                gpus.append(GpuInfo(name=name, backend="ROCm"))
        return gpus
    except (OSError, subprocess.SubprocessError):
        return []


def _detect_apple_gpu() -> list[GpuInfo]:
    if platform.system() != "Darwin":
        return []
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        for line in out.splitlines():
            if "Chip" in line or "Model" in line:
                name = line.split(":")[-1].strip()
                if name:
                    return [GpuInfo(name=name, backend="Metal")]
    except (OSError, subprocess.SubprocessError):
        pass
    return []


def _vendor_to_backend(vendor: str) -> str:
    v = vendor.lower()
    if "nvidia" in v:
        return "CUDA"
    if "advanced micro devices" in v or "amd" in v or "ati" in v:
        return "ROCm"
    if "intel" in v:
        return "Vulkan"
    if "apple" in v:
        return "Metal"
    return "GPU"


def _read_sysfs_vram_for_pci(pci_addr: str) -> float | None:
    """Return VRAM in GB by matching pci_addr ('0a:00.0') to /sys/class/drm/card*/device."""
    try:
        cards = list(Path("/sys/class/drm").glob("card[0-9]*"))
    except OSError:
        return None
    target = f"0000:{pci_addr}"
    for card in cards:
        try:
            link = card.resolve()
        except OSError:
            continue
        if target not in str(link):
            continue
        vram_file = card / "device" / "mem_info_vram_total"
        try:
            bytes_total = int(vram_file.read_text().strip())
        except (OSError, ValueError):
            return None
        if bytes_total > 0:
            return bytes_total / (1024**3)
    return None


def _detect_gpus_via_lspci() -> list[GpuInfo]:
    """Universal fallback for any Linux box with lspci. No driver tooling required."""
    try:
        out = subprocess.check_output(
            ["lspci", "-mm"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    gpus: list[GpuInfo] = []
    classes = ('"VGA compatible controller"', '"3D controller"', '"Display controller"')
    for line in out.splitlines():
        if not any(c in line for c in classes):
            continue
        parts = re.findall(r'"([^"]*)"', line)
        if len(parts) < 3:
            continue
        # `lspci -mm` line: BUS:DEV.FN "Class" "Vendor" "Device" "SVendor" "SDevice" ...
        # The BUS:DEV.FN prefix is unquoted, so class=parts[0], vendor=parts[1], device=parts[2].
        pci_addr = line.split(maxsplit=1)[0]
        vendor, device = parts[1], parts[2]
        # Friendly name: vendor short + product description
        vendor_short = vendor.split()[0] if vendor else ""
        if vendor_short.lower() in ("advanced", "ati"):
            vendor_short = "AMD"
        name = f"{vendor_short} {device}".strip()
        gpus.append(
            GpuInfo(
                name=name,
                vram_gb=_read_sysfs_vram_for_pci(pci_addr),
                backend=_vendor_to_backend(vendor),
            )
        )
    return gpus


def collect() -> SystemInfo:
    import psutil

    cpu_model = _detect_cpu_model()
    cpu_cores = psutil.cpu_count(logical=True) or 1
    cpu_physical_cores = psutil.cpu_count(logical=False) or 0
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024**3)
    available_ram_gb = mem.available / (1024**3)
    os_name = f"{platform.system()} {platform.release()}"

    gpus = (
        _detect_nvidia_gpus()
        or _detect_amd_gpus()
        or _detect_apple_gpu()
        or _detect_gpus_via_lspci()
    )

    return SystemInfo(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        cpu_physical_cores=cpu_physical_cores,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        os=os_name,
        gpus=gpus,
    )
