"""
Config Manager - Loads YAML config + .env variables
Replaces all hardcoded paths and settings from the original notebook
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

# Load .env file
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]


class Config:
    """Central configuration manager for the entire pipeline."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = ROOT_DIR / "configs" / "config.yaml"

        self._config = self._load_yaml(config_path)
        self._resolve_env_vars(self._config)
        logger.info(f"✅ Config loaded from: {config_path}")

    def _load_yaml(self, path: Path) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_env_vars(self, obj: Any):
        """Replace ${VAR} placeholders with actual env variables."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    obj[key] = os.getenv(env_var, value)
                else:
                    self._resolve_env_vars(value)
        elif isinstance(obj, list):
            for item in obj:
                self._resolve_env_vars(item)

    def get(self, *keys, default=None):
        """Get nested config value using dot-notation keys."""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value

    # ── Convenience properties ──────────────────────────────────────────

    @property
    def project(self) -> dict:
        return self._config["project"]

    @property
    def data(self) -> dict:
        cfg = self._config["data"]
        # Resolve all paths relative to project root
        for key in ["raw_dir", "processed_dir", "delta_dir"]:
            cfg[key] = str(ROOT_DIR / cfg[key])
        return cfg

    @property
    def preprocessing(self) -> dict:
        return self._config["preprocessing"]

    @property
    def training(self) -> dict:
        return self._config["training"]

    @property
    def mlflow(self) -> dict:
        cfg = self._config["mlflow"]
        cfg["tracking_uri"] = str(ROOT_DIR / cfg["tracking_uri"])
        return cfg

    @property
    def serving(self) -> dict:
        return self._config["serving"]

    @property
    def azure(self) -> dict:
        return self._config["azure"]

    @property
    def logging(self) -> dict:
        return self._config["logging"]

    @property
    def root_dir(self) -> Path:
        return ROOT_DIR


# Singleton instance
_config_instance = None


def get_config(config_path: str = None) -> Config:
    """Get or create the singleton Config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
