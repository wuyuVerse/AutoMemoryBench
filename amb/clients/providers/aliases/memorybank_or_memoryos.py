"""Method-identity alias for the official MemoryOS client factory."""

from __future__ import annotations

from amb.clients.providers.local_sources.memoryos import MemoryOSOfficialSourceClient, create_client

__all__ = ["MemoryOSOfficialSourceClient", "create_client"]
