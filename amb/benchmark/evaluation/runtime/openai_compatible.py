"""OpenAI-compatible chat client for foundation-model protocol evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import sys
import time
from typing import Any
from urllib import error, request


sys.modules.setdefault("amb.benchmark.evaluation.openai_compatible", sys.modules[__name__])
sys.modules.setdefault("agent_memory_benchmark.evaluation.openai_compatible", sys.modules[__name__])


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    usage: dict[str, Any]
    raw: dict[str, Any]


class OpenAICompatibleChatClient:
    """Small OpenAI-compatible chat client with JSON-response fallback."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_s: float = 60.0,
        max_attempts: int | None = None,
        retry_base_sleep_s: float | None = None,
        retry_sleep_cap_s: float | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        resolved_max_attempts = _int_env("AMB_OPENAI_COMPAT_MAX_ATTEMPTS", max_attempts, default=12)
        resolved_base_sleep = _float_env("AMB_OPENAI_COMPAT_RETRY_BASE_SLEEP_S", retry_base_sleep_s, default=2.0)
        resolved_sleep_cap = _float_env("AMB_OPENAI_COMPAT_RETRY_SLEEP_CAP_S", retry_sleep_cap_s, default=60.0)
        if resolved_max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if resolved_base_sleep <= 0:
            raise ValueError("retry_base_sleep_s must be positive")
        if resolved_sleep_cap <= 0:
            raise ValueError("retry_sleep_cap_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_attempts = resolved_max_attempts
        self.retry_base_sleep_s = resolved_base_sleep
        self.retry_sleep_cap_s = resolved_sleep_cap

    def create_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        require_json: bool = True,
    ) -> ChatCompletion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if require_json:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self._post_json_with_retry("/chat/completions", payload)
        except error.HTTPError as exc:
            # Some OpenAI-compatible providers reject `response_format`.
            if require_json and exc.code in {400, 404, 422}:
                payload.pop("response_format", None)
                data = self._post_json_with_retry("/chat/completions", payload)
            else:
                raise
        content = _message_content(data)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ChatCompletion(content=content, usage=usage, raw=data)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json_with_retry(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._post_json(path, payload)
            except error.HTTPError as exc:
                last_exc = exc
                if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt == self.max_attempts:
                    raise
            except (TimeoutError, error.URLError) as exc:
                last_exc = exc
                if attempt == self.max_attempts:
                    raise
            time.sleep(min(self.retry_base_sleep_s * (2 ** (attempt - 1)), self.retry_sleep_cap_s))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unreachable retry state")


def parse_json_response(content: str) -> dict[str, Any]:
    """Best-effort JSON parsing for model outputs."""

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"response": text}
    except json.JSONDecodeError:
        start = text.find("{")
        end = _balanced_json_end(text, start)
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end])
                return data if isinstance(data, dict) else {"response": text}
            except json.JSONDecodeError:
                pass
        if start != -1:
            repaired = _autoclose_json_fragment(text[start:])
            if repaired is not None:
                try:
                    data = json.loads(repaired)
                    return data if isinstance(data, dict) else {"response": text}
                except json.JSONDecodeError:
                    pass
    return {"response": content}


def _balanced_json_end(text: str, start: int) -> int:
    if start < 0 or start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _autoclose_json_fragment(fragment: str) -> str | None:
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in fragment:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch == "}":
            if not stack or stack[-1] != "{":
                return None
            stack.pop()
            continue
        if ch == "]":
            if not stack or stack[-1] != "[":
                return None
            stack.pop()
    if in_string:
        return None
    closers = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return fragment + closers


def _message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def _int_env(name: str, value: int | None, *, default: int) -> int:
    if value is not None:
        return int(value)
    raw = os.getenv(name)
    if raw:
        return int(raw)
    return default


def _float_env(name: str, value: float | None, *, default: float) -> float:
    if value is not None:
        return float(value)
    raw = os.getenv(name)
    if raw:
        return float(raw)
    return default
