import json
from typing import Any, Dict
from plugins.core.hardware_detector import HardwareDetector

class HardwareFallbackCascade:
    """Selects a hardware config preset based on detected system hardware.

    This class implements a cascading logic to choose the best available
    hardware configuration. It inspects the hardware profile provided by the
    HardwareDetector and selects a corresponding configuration from a
    predefined set of fallbacks (e.g., high-end GPU, MPS, CPU).
    """
    def __init__(self, fallback_config: str = "configs/training/hardware_fallback.json"):
        """Initializes the HardwareFallbackCascade.

        Args:
            fallback_config: Path to the JSON file containing the hardware
                fallback presets.
        """
        with open(fallback_config) as f:
            self.fallbacks = json.load(f)
        self.detector = HardwareDetector()

    def get_cascaded_config(self, primary_profile: str = "gpu_high") -> Dict[str, Any]:
        """Gets the optimal configuration based on a cascading hardware check.

        It checks for hardware in a descending order of preference (e.g.,
        high-memory GPU, standard GPU, Apple MPS, CPU) and returns the
        configuration for the first match found.

        Args:
            primary_profile: The desired primary profile (e.g., 'gpu_high').
                This is currently unused but preserved for future enhancements.

        Returns:
            A dictionary containing the selected hardware configuration preset.
        """
        profile = self.detector.hardware_profile
        if profile.get("cuda_available") and profile.get("cuda_device_count", 0) > 0:
            # Simple check for now, can be expanded for multi-gpu tiers
            if profile.get("gpu_details", [{}])[0].get("memory_gb", 0) > 20:
                # Using ascend_npu as a stand-in for high-end GPU
                return self.fallbacks.get("ascend_npu", {})
            else:
                # Using mps as a stand-in for mid-range GPU
                return self.fallbacks.get("mps", {})
        elif profile.get("mps_available"):
            return self.fallbacks.get("mps", {})
        else:
            return self.fallbacks.get("cpu", {})

    def apply_to_wrapper(self, wrapper: Any):
        """Applies the cascaded configuration to a given wrapper instance.

        This is a convenience method that retrieves the optimal hardware
        configuration and directly updates the configuration dictionary of a
        compatible wrapper object.

        Args:
            wrapper: An object that has a `cfg` attribute (a dictionary) to be updated.
        """
        config = self.get_cascaded_config()
        # Assumes the wrapper object has a `cfg` attribute that is a dictionary.
        if hasattr(wrapper, 'cfg') and isinstance(wrapper.cfg, dict):
            wrapper.cfg.update(config)