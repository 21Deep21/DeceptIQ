"""Shared configuration loading and application logging.

Secrets are NEVER stored in config.yaml - they come from environment
variables only. URLs are sanitized by callers before logging; no
secrets are ever written to logs.
"""

from __future__ import annotations

import functools
import logging
import pathlib
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).resolve().parent / "config.yaml"

_APP_LOGGING_CONFIGURED = False


@functools.lru_cache(maxsize=4)
def get_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and cache config.yaml (cached per path)."""
    p = pathlib.Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"configuration file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.debug("loaded configuration from %s", p)
    return cfg


def reset_config_cache() -> None:
    """Drop cached configuration (used by tests)."""
    get_config.cache_clear()


def setup_console_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_app_logging(cfg: Dict[str, Any]) -> None:
    """Configure ROOT logging: rotating file + console (idempotent).

    Rotating file lives at <log_dir>/<logging.file> (default
    logs/app.log). Records startup, model loading, prediction events,
    API and application errors. Callers must sanitize URLs (strip
    credentials) before logging them.
    """
    global _APP_LOGGING_CONFIGURED
    if _APP_LOGGING_CONFIGURED:
        return
    log_cfg = cfg.get("logging", {})
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = pathlib.Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / str(log_cfg.get("file", "app.log"))
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(level)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=int(log_cfg.get("max_bytes", 5_242_880)),
        backupCount=int(log_cfg.get("backup_count", 3)),
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(file_handler)
    root.addHandler(console)
    _APP_LOGGING_CONFIGURED = True
    logger.info("application logging configured: file=%s level=%s", log_file, level_name)


def load_env_file(path: Optional[str] = None) -> Dict[str, str]:
    """Load KEY=VALUE pairs from a git-ignored .env file into os.environ.

    Variables already present in the real environment are NOT overridden.
    Only the KEY NAMES are ever logged - values (secrets) never are.
    Secrets therefore stay out of config.yaml and out of source code.
    """
    import os

    p = pathlib.Path(path) if path else (pathlib.Path(__file__).resolve().parent / ".env")
    loaded: Dict[str, str] = {}
    if not p.exists():
        return loaded
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    if loaded:
        logger.info("loaded environment variables from %s: %s", p, sorted(loaded))
    return loaded
