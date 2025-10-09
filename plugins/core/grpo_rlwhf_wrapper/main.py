"""Utility helpers that map AI-RLWHF presets to ms-swift GRPO launch commands."""
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

try:  # Prefer production wrapper when available
    from plugins.core.grpo_production_wrapper import ProductionGRPOWrapper

    __all__ = ["ProductionGRPOWrapper"]
except Exception:  # pragma: no cover
    ProductionGRPOWrapper = None  # type: ignore

from plugins.core.custom_honesty_rm import collect_reward_config, custom_honesty_reward_entrypoint
from scripts.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = Path("configs/transformer-lab/grpo_config.yaml")


@dataclass
class LaunchBundle:
    """A container for the environment, arguments, and command for a training job.

    Attributes:
        env: A dictionary of environment variables to set for the job.
        args: A dictionary of command-line arguments for the training script.
        command: The fully rendered command string to be executed.
    """
    env: Dict[str, str]
    args: Dict[str, Any]
    command: str


def _stringify(value: Any) -> str:
    """Serializes a value into a string suitable for command-line arguments."""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class GRPORLWHFWrapper:
    """Creates reusable launch bundles for ms-swift GRPO experiments.

    This class reads a YAML configuration file to generate the necessary
    environment variables, command-line arguments, and the final command
    string required to launch a GRPO training job using ms-swift.
    """

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        """Initializes the GRPORLWHFWrapper.

        Args:
            config_path: The path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path, default={})

    def ensure_reward_artifact(self) -> Dict[str, Any]:
        """Ensures the custom honesty reward model artifact is created.

        This method reads configuration details for the reward model plugin,
        invokes its entrypoint to generate the reward model artifact, and
        returns a payload containing the resulting configuration.

        Returns:
            A dictionary payload with the reward configuration.
        """
        pipeline = self.config.get("pipeline", {})
        reward_params = pipeline.get("reward_plugin_params", {})
        reward_overrides = {
            "dataset_path": pipeline.get("preprocess_args", {}).get("output_path"),
            "output_dir": self.config.get("paths", {}).get("reward_model_cache"),
            **reward_params,
        }
        reward_config = collect_reward_config(reward_overrides)
        payload = custom_honesty_reward_entrypoint(**reward_overrides)
        payload["reward_config"] = reward_config
        return payload

    def build_launch_bundle(self, hardware_profile: str = "single_gpu") -> LaunchBundle:
        """Constructs the full launch bundle for a given hardware profile.

        Args:
            hardware_profile: The name of the hardware profile to use, as
                defined in the configuration file (e.g., 'single_gpu').

        Returns:
            A LaunchBundle instance containing the environment, arguments,
            and command for the training job.
        """
        ms_swift_cfg = self.config.get("ms_swift", {})
        hardware_profiles = self.config.get("hardware_profiles", {})
        profile = hardware_profiles.get(hardware_profile, hardware_profiles.get("single_gpu", {}))
        env = dict(ms_swift_cfg.get("launch_env", {}))
        env.setdefault("UNSLOTH_VLLM_STANDBY", "1")
        args: Dict[str, Any] = dict(ms_swift_cfg.get("grpo_args", {}))
        args.setdefault("reward_module", ms_swift_cfg.get("reward_module", "plugins.core.custom_honesty_rm"))
        args.setdefault("hardware_profile", hardware_profile)
        args.update({f"hardware_{key}": value for key, value in profile.items()})
        command = self._render_command(ms_swift_cfg, args)
        return LaunchBundle(env=env, args=args, command=command)

    def _render_command(self, ms_swift_cfg: Mapping[str, Any], args: Mapping[str, Any]) -> str:
        """Renders the final shell command for launching the training job.

        Args:
            ms_swift_cfg: The ms-swift configuration dictionary.
            args: The command-line arguments dictionary.

        Returns:
            The fully-formed command string.
        """
        entry = ms_swift_cfg.get("trainer_entry", "swift.llm.train.run_grpo")
        serialized_args = " ".join(f"--{key.replace('_', '-')}" f" {shlex.quote(_stringify(value))}" for key, value in args.items())
        return f"python3 -m {entry} {serialized_args}"


def load_launch_bundle(hardware_profile: str = "single_gpu", config_path: Path | str = DEFAULT_CONFIG_PATH) -> LaunchBundle:
    """Loads a launch bundle for a given hardware profile.

    This is a convenience function that instantiates the GRPORLWHFWrapper,
    ensures the reward artifact is created, and returns the launch bundle.

    Args:
        hardware_profile: The hardware profile to use for the launch.
        config_path: The path to the main configuration file.

    Returns:
        A configured LaunchBundle instance.
    """
    wrapper = GRPORLWHFWrapper(config_path)
    wrapper.ensure_reward_artifact()
    return wrapper.build_launch_bundle(hardware_profile=hardware_profile)
