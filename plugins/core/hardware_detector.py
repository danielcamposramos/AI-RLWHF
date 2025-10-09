"""Hardware detection utilities for adaptive plugin behaviour."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

try:  # Optional torch import
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class HardwareDetector:
    """Collect a normalized hardware profile used by wrappers and plugins."""

    def __init__(self) -> None:
        self.hardware_profile = self._detect_hardware()

    def _detect_hardware(self) -> Dict[str, Any]:
        profile: Dict[str, Any] = {
            "system": platform.system(),
            "cpu_count": os.cpu_count() or 1,
            "cuda_available": bool(torch and torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch and torch.cuda.is_available() else 0,
            "mps_available": bool(torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),  # type: ignore[attr-defined]
            "npu_available": self._detect_ascend_npu(),
            "total_memory_gb": 0.0,
            "gpu_details": [],
        }
        if profile["cuda_available"]:
            assert torch is not None
            for idx in range(profile["cuda_device_count"]):
                props = torch.cuda.get_device_properties(idx)
                memory_gb = float(props.total_memory) / (1024**3)
                profile["gpu_details"].append(
                    {
                        "name": props.name,
                        "memory_gb": memory_gb,
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
                profile["total_memory_gb"] += memory_gb
        return profile

    def _detect_ascend_npu(self) -> bool:
        npu_cli = shutil.which("npu-smi")
        if not npu_cli:
            return False
        try:
            proc = subprocess.run([npu_cli, "info"], check=False, capture_output=True, text=True)
        except OSError:
            return False
        return proc.returncode == 0

    def get_recommended_batch_size(self, model_size: str = "medium") -> int:
        profile = self.hardware_profile
        if profile["cuda_available"]:
            total_mem = profile["total_memory_gb"]
            if model_size == "large":
                if total_mem >= 40:
                    return 8
                if total_mem >= 24:
                    return 4
                if total_mem >= 16:
                    return 2
                return 1
            if total_mem >= 40:
                return 16
            if total_mem >= 24:
                return 8
            if total_mem >= 16:
                return 4
            return 2
        if profile["mps_available"]:
            return 2
        return 1

    def serialize(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.hardware_profile, indent=2), encoding="utf-8")


__all__ = ["HardwareDetector"]
