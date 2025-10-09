import torch
import platform
import subprocess
import json
import logging
import os
from typing import Dict, Any

logger = logging.getLogger("hardware_detector")

class HardwareDetector:
    """Detects and profiles the hardware available on the system.

    This class provides a comprehensive overview of the system's hardware,
    including CPU, CUDA-enabled GPUs, Apple MPS, and Huawei Ascend NPUs.
    The profile can be used to dynamically configure other parts of the
    application, such as plugin loaders and training wrappers.

    Attributes:
        hardware_profile: A dictionary containing detailed information about
            the detected hardware.
    """

    def __init__(self):
        """Initializes the HardwareDetector and profiles the system."""
        self.hardware_profile = self._detect_hardware()

    def _detect_hardware(self) -> Dict[str, Any]:
        """Performs a comprehensive detection of system hardware.

        Returns:
            A dictionary containing a detailed hardware profile.
        """
        profile = {
            "system": platform.system(),
            "cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": 0,
            "mps_available": False,
            "npu_available": False,
            "total_memory_gb": 0.0,
            "gpu_details": []
        }

        if torch.cuda.is_available():
            profile["cuda_device_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "name": gpu_props.name,
                    "memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                }
                profile["gpu_details"].append(gpu_info)
                profile["total_memory_gb"] += gpu_info["memory_gb"]

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            profile["mps_available"] = True

        profile["npu_available"] = self._detect_ascend_npu()

        logger.info(f"Detected hardware profile: {profile}")
        return profile

    def _detect_ascend_npu(self) -> bool:
        """Detects the presence of a Huawei Ascend NPU.

        This is done by attempting to run the `npu-smi` command-line tool.

        Returns:
            True if an NPU is detected, False otherwise.
        """
        try:
            result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=False)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

    def get_recommended_batch_size(self, model_size: str = "medium") -> int:
        """Recommends a training batch size based on hardware and model size.

        Args:
            model_size: The size of the model being trained, used as a hint
                for memory requirements. Expected values are 'small',
                'medium', or 'large'.

        Returns:
            A recommended integer batch size.
        """
        if not self.hardware_profile.get("cuda_available"):
            return 1  # Conservative default for CPU/MPS

        total_memory = self.hardware_profile.get("total_memory_gb", 0)

        if model_size == "large":  # 7B+ models
            if total_memory > 40: return 8
            if total_memory > 24: return 4
            if total_memory > 16: return 2
            return 1
        # Medium/small models
        if total_memory > 40: return 16
        if total_memory > 24: return 8
        if total_memory > 16: return 4
        return 2