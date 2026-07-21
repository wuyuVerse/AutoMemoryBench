"""Environment-variable validation helpers."""

from __future__ import annotations


_ENV_NAME_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"


def is_env_var_name(name: str) -> bool:
    """Return true when ``name`` is an uppercase shell-style environment variable."""
    return bool(name) and name[0].isalpha() and all(ch in _ENV_NAME_CHARS for ch in name)


def require_env_var_name(name: str, *, option_name: str = "api-key-env") -> None:
    """Reject direct secret-looking values without echoing the rejected input."""
    if not is_env_var_name(name):
        raise SystemExit(f"{option_name} must be an environment variable name")
