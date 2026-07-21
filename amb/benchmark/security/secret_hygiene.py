"""Secret-hygiene checks for benchmark configuration payloads."""

from __future__ import annotations

import re
from typing import Any

from amb.benchmark.security.env_vars import is_env_var_name


SECRET_KEY_RE = re.compile(r"(api[_-]?key|access[_-]?token|secret|password|bearer|authorization)", re.I)
SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|AKIA[0-9A-Z]{12,}|"
    r"AIza[0-9A-Za-z_-]{16,}|[A-Za-z0-9+/]{32,}={0,2})"
)


def secret_like_paths(payload: Any, *, path: str = "$") -> list[str]:
    """Return JSON paths that look like materialized secrets.

    Environment-variable reference fields such as ``api_key_env`` and
    ``credential_source`` are allowed when their values are env var names. The
    function never reads environment values; it only checks serialized config
    payloads and CLI-provided loader kwargs.
    """

    hits: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if _is_env_reference_key(key_text):
                if isinstance(value, str) and not is_env_var_name(value):
                    hits.append(child_path)
                continue
            if SECRET_KEY_RE.search(key_text) and isinstance(value, str) and value:
                hits.append(child_path)
                continue
            hits.extend(secret_like_paths(value, path=child_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            hits.extend(secret_like_paths(value, path=f"{path}[{index}]"))
    elif isinstance(payload, str) and SECRET_VALUE_RE.search(payload):
        hits.append(path)
    return hits


def _is_env_reference_key(key: str) -> bool:
    normalized = key.lower()
    return normalized == "credential_source" or normalized.endswith("_env") or normalized.endswith("_env_var")
