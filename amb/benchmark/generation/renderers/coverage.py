"""Renderer coverage metadata."""

from __future__ import annotations

from amb.benchmark.generation.renderers.adversarial import ADVERSARIAL_EVENT_TYPES
from amb.benchmark.generation.renderers.document import DOCUMENT_EVENT_TYPES
from amb.benchmark.generation.renderers.platform import PLATFORM_EVENT_TYPES
from amb.benchmark.generation.renderers.tool import TOOL_EVENT_TYPES

RENDERER_EVENT_TYPES: dict[str, tuple[str, ...]] = {
    "adversarial": ADVERSARIAL_EVENT_TYPES,
    "document": DOCUMENT_EVENT_TYPES,
    "platform": PLATFORM_EVENT_TYPES,
    "tool": TOOL_EVENT_TYPES,
}
