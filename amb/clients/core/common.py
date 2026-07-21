"""Shared helpers for real AutoMemoryBench memory-system client factories."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(raw: str | None, default: Path) -> Path:
    if not raw:
        return default
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_site_packages_from_venv(venv_root: str | Path | None) -> Path | None:
    if not venv_root:
        return None
    root = Path(venv_root)
    site_packages = venv_site_packages(root)
    if site_packages is None:
        raise FileNotFoundError(f"virtualenv site-packages not found under {root}")
    raw_path = str(site_packages)
    if raw_path not in sys.path:
        sys.path.insert(0, raw_path)
    return site_packages


def ensure_source_path(source_root: str | Path) -> Path:
    """Add an official source tree or its src/ directory to sys.path."""

    root = resolve_path(str(source_root), PROJECT_ROOT / str(source_root))
    if not root.exists():
        raise FileNotFoundError(f"official source path not found: {root}")
    candidate = root / "src"
    import_root = candidate if candidate.exists() else root
    raw_path = str(import_root)
    if raw_path not in sys.path:
        sys.path.insert(0, raw_path)
    return import_root


def venv_site_packages(venv_root: Path) -> Path | None:
    candidates = sorted((venv_root / "lib").glob("python*/site-packages"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def import_or_raise(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"required module {module_name!r} is not importable; install it into the configured venv first"
        ) from exc


def effective_api_key(
    *,
    api_key: str | None,
    api_key_env: str,
    fallback_envs: tuple[str, ...] = (),
) -> str | None:
    if api_key:
        return api_key
    value = os.getenv(api_key_env)
    if value:
        return value
    for name in fallback_envs:
        value = os.getenv(name)
        if value:
            return value
    return None
