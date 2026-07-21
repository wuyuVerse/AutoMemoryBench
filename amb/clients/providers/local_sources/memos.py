"""Local official-source MemOS client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv, ensure_source_path, resolve_path


def _inject_openai_key(cfg: dict) -> None:
    """Inject OPENAI_API_KEY (env) into MemOS openai/universal_api backend configs.
    Keys live only in env (configs/*.bash); never hardcoded in the committed JSON."""
    import os
    key = os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        return
    def walk(node):
        if isinstance(node, dict):
            be = node.get("backend")
            c = node.get("config")
            if be in ("openai", "openai_new", "universal_api", "azure", "qwen", "deepseek") and isinstance(c, dict):
                c.setdefault("api_key", key)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(cfg)


class MemOSOfficialSourceClient:
    """Thin adapter over the official ``memos.mem_os.core.MOS`` API."""

    def __init__(
        self,
        mos: Any,
        *,
        mem_cube_id: str | None = None,
        user_id: str | None = None,
        default_limit: int = 5,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.mos = mos
        self.mem_cube_id = mem_cube_id
        self.user_id = user_id or str(getattr(mos, "user_id", "") or "")
        self.default_limit = default_limit
        self.case_id: str | None = None

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        if self.mem_cube_id:
            self.mos.delete_all(mem_cube_id=self.mem_cube_id, user_id=self._target_user_id())
        return {"ok": True}

    def add(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        payload = messages if messages is not None else _content_from_payload(content)
        result = self.mos.add(
            messages=payload if messages is not None else None,
            memory_content=None if messages is not None else str(payload),
            mem_cube_id=self.mem_cube_id,
            user_id=self._target_user_id(),
            session_id=str((metadata or {}).get("case_id") or self.case_id or ""),
        )
        return {"result": result, "user_id": self._target_user_id(), "case_id": self.case_id}

    def search(
        self,
        query: str | None = None,
        *,
        user_id: str | None = None,
        limit: int | None = None,
        top_k: int | None = None,
        **_: Any,
    ) -> Any:
        if not query:
            raise ValueError("query is required for MemOS search")
        k = int(limit or top_k or self.default_limit)
        return self.mos.search(str(query), user_id=self._target_user_id(), install_cube_ids=None, top_k=k)

    def get_all(self, *, user_id: str | None = None, **_: Any) -> Any:
        return self.mos.get_all(mem_cube_id=self.mem_cube_id, user_id=self._target_user_id())

    def delete(self, memory_id: str | None = None, *, user_id: str | None = None, **_: Any) -> Any:
        if not memory_id:
            raise ValueError("memory_id is required for MemOS delete")
        if not self.mem_cube_id:
            raise ValueError("mem_cube_id is required for MemOS per-memory delete")
        return self.mos.delete(self.mem_cube_id, str(memory_id), user_id=self._target_user_id())

    def _target_user_id(self) -> str:
        if not self.user_id:
            raise ValueError("MemOS adapter requires a configured registered user_id")
        return self.user_id


def create_client(
    *,
    source_root: str = "related_work/repos/MemOS",
    venv_root: str | None = None,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    mem_cube_id: str | None = None,
    mem_cube_config_path: str | None = None,
    mem_cube_dump_path: str | None = None,
    auto_register_mem_cube: bool = False,
    user_id: str | None = None,
    default_limit: int = 5,
) -> MemOSOfficialSourceClient:
    """Create a client from the locally cloned official MemOS source tree."""

    if venv_root:
        ensure_site_packages_from_venv(Path(venv_root))
    ensure_source_path(source_root)
    # MemOS keeps a single global user-manager sqlite at MEMOS_DIR/memos_users.db,
    # where MEMOS_DIR defaults to <MEMOS_BASE_PATH or cwd>/.memos. On the cluster every
    # shard's cwd is the shared repo root, so all concurrent shards write ONE users.db
    # -> "attempt to write a readonly database" lock contention (same class as
    # simplemem's CWD-relative LANCEDB_PATH). Give each process an isolated MEMOS_DIR
    # in pod-local /tmp before MemOS reads settings, so every shard gets its own db.
    if not os.environ.get("MEMOS_BASE_PATH"):
        import tempfile
        os.environ["MEMOS_BASE_PATH"] = tempfile.mkdtemp(prefix="memos_base_")
    # MEMOS_DIR is a module-level constant evaluated at `import memos.settings`. If
    # settings was already imported in this process the constant is frozen against the
    # old (shared) path, so patch it directly to the isolated base too.
    try:
        import memos.settings as _memos_settings  # type: ignore
        _memos_settings.MEMOS_DIR = Path(os.environ["MEMOS_BASE_PATH"]) / ".memos"
        _memos_settings.MEMOS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    from memos.mem_os.main import MOS  # type: ignore
    from memos.configs.mem_os import MOSConfig  # type: ignore

    if config_path:
        import json as _json
        cfg_dict = _json.loads(Path(config_path).read_text())
        _inject_openai_key(cfg_dict)
        mos_config = MOSConfig(**cfg_dict)
    elif config is not None:
        _inject_openai_key(config)
        mos_config = MOSConfig(**config)
    else:
        raise ValueError("MemOS official adapter requires config or config_path for MOSConfig")
    target_user_id = user_id or str(getattr(mos_config, "user_id", "") or "")
    if target_user_id:
        if getattr(mos_config, "user_id", None) != target_user_id:
            mos_config.user_id = target_user_id
        _ensure_user_manager_user_exists(target_user_id)
    mos = MOS(mos_config)
    if auto_register_mem_cube:
        if not mem_cube_id:
            raise ValueError("mem_cube_id is required when auto_register_mem_cube=True")
        if not mem_cube_config_path:
            raise ValueError("mem_cube_config_path is required when auto_register_mem_cube=True")
        if not mem_cube_dump_path:
            raise ValueError("mem_cube_dump_path is required when auto_register_mem_cube=True")
        _register_mem_cube_from_config(
            mos,
            mem_cube_id=mem_cube_id,
            mem_cube_config_path=mem_cube_config_path,
            mem_cube_dump_path=mem_cube_dump_path,
            user_id=target_user_id or str(getattr(mos, "user_id", "") or ""),
        )
    return MemOSOfficialSourceClient(
        mos,
        mem_cube_id=mem_cube_id,
        user_id=target_user_id or str(getattr(mos, "user_id", "") or ""),
        default_limit=default_limit,
    )


def _content_from_payload(content: Any) -> str:
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)


def _register_mem_cube_from_config(
    mos: Any,
    *,
    mem_cube_id: str,
    mem_cube_config_path: str,
    mem_cube_dump_path: str,
    user_id: str,
) -> None:
    from memos.configs.mem_cube import GeneralMemCubeConfig  # type: ignore
    from memos.mem_cube.general import GeneralMemCube  # type: ignore

    if not user_id:
        raise ValueError("user_id is required to register a MemOS MemCube")
    _ensure_user_exists(mos, user_id)
    config_path = resolve_path(mem_cube_config_path, Path(mem_cube_config_path))
    dump_path = resolve_path(mem_cube_dump_path, Path(mem_cube_dump_path))
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    if not dump_path.exists() or not any(dump_path.iterdir()):
        dump_path.mkdir(parents=True, exist_ok=True)
        import json as _json
        _cube_dict = _json.loads(Path(config_path).read_text())
        _inject_openai_key(_cube_dict)
        cube_config = GeneralMemCubeConfig.model_validate(_cube_dict)
        cube_config.user_id = user_id
        cube_config.cube_id = mem_cube_id
        GeneralMemCube(cube_config).dump(str(dump_path))
    mos.register_mem_cube(str(dump_path), mem_cube_id=mem_cube_id, user_id=user_id)


def _ensure_user_exists(mos: Any, user_id: str) -> None:
    try:
        if any(row.get("user_id") == user_id and row.get("is_active", True) for row in mos.list_users()):
            return
    except Exception:
        pass
    created_user_id = mos.create_user(user_id=user_id, user_name=user_id)
    if created_user_id and str(created_user_id) != user_id:
        raise ValueError(f"MemOS created unexpected user_id {created_user_id!r}; expected {user_id!r}")


def _ensure_user_manager_user_exists(user_id: str) -> None:
    from memos.mem_user.user_manager import UserManager  # type: ignore

    # MOS creates its own UserManager during initialization and validates config.user_id.
    # Creating the user through the official manager first keeps scoped benchmark users valid.
    UserManager(user_id=user_id)
