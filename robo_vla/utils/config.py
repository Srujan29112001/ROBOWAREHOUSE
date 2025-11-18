"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.

    Returns:
        Configuration dictionary
    """
    load_dotenv()  # Load environment variables

    if config_path is None:
        config_path = os.path.join(
            Path(__file__).parent.parent,
            "configs",
            "config.yaml"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with environment variables
    config = _override_with_env(config)

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def _override_with_env(config: Dict[str, Any], prefix: str = "ROBOVLA") -> Dict[str, Any]:
    """Override config values with environment variables."""
    for key, value in config.items():
        env_key = f"{prefix}_{key.upper()}"

        if isinstance(value, dict):
            config[key] = _override_with_env(value, prefix=env_key)
        elif env_key in os.environ:
            # Type casting based on original value
            if isinstance(value, bool):
                config[key] = os.environ[env_key].lower() in ("true", "1", "yes")
            elif isinstance(value, int):
                config[key] = int(os.environ[env_key])
            elif isinstance(value, float):
                config[key] = float(os.environ[env_key])
            else:
                config[key] = os.environ[env_key]

    return config


def get_model_cache_dir() -> str:
    """Get directory for cached models."""
    cache_dir = os.environ.get("MODEL_CACHE_DIR", "./models")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_data_dir() -> str:
    """Get directory for data storage."""
    data_dir = os.environ.get("DATA_DIR", "./data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir
