"""Hardware detection: CPU, RAM, GPU."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import dataclass, field


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
    gpus: list[GpuInfo] = field(default_factory=list)
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
        import cpuinfo  # type: ignore[import-untyped]

        info = cpuinfo.get_cpu_info()
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
        gpus = []
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
        gpus = []
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


def collect() -> SystemInfo:
    import psutil

    cpu_model = _detect_cpu_model()
    cpu_cores = psutil.cpu_count(logical=True) or 1
    cpu_physical_cores = psutil.cpu_count(logical=False) or 0
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024**3)
    available_ram_gb = mem.available / (1024**3)
    os_name = f"{platform.system()} {platform.release()}"

    gpus = _detect_nvidia_gpus() or _detect_amd_gpus() or _detect_apple_gpu()

    return SystemInfo(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        cpu_physical_cores=cpu_physical_cores,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        os=os_name,
        gpus=gpus,
    )
