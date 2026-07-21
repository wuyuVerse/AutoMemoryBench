"""Worker process entry — runs INSIDE a system's own venv.

Loads the system's agent (memory integration or agent framework) from its config
and serves a line-oriented JSON-RPC over stdin/stdout. One request per line:

    {"op": "reset", "case_id": "..."}            -> {"ok": true}
    {"op": "observe", "observation": {...}}      -> {"ok": true}
    {"op": "answer_or_act", "request": {...}}    -> {"ok": true, "result": {...}}
    {"op": "export_trace"}                       -> {"ok": true, "result": {...}|null}
    {"op": "close"}                              -> {"ok": true}  (then exit)

All payloads are JSON values — the same dicts the in-process runner passes, so the
worker is behaviourally transparent. Only this module runs in the dirty venv; it
imports amb (SDK-free at module top-level) plus, lazily, the system's SDK.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback


def _load_agent(config_path: str):
    # Reuse the CLI's loader so memory-integration and agent configs dispatch
    # exactly as they do today (same metadata, execution-mode handling).
    from amb.benchmark.interfaces.commands.evaluation import _load_cli_agent

    return _load_cli_agent(config_path).agent


def _handle(agent, msg: dict) -> dict:
    op = msg.get("op")
    if op == "reset":
        agent.reset(msg["case_id"])
        return {"ok": True}
    if op == "observe":
        agent.observe(msg["observation"])
        return {"ok": True}
    if op == "answer_or_act":
        return {"ok": True, "result": agent.answer_or_act(msg["request"])}
    if op == "export_trace":
        fn = getattr(agent, "export_trace", None)
        return {"ok": True, "result": fn() if callable(fn) else None}
    if op == "ping":
        return {"ok": True, "result": "pong"}
    if op == "close":
        return {"ok": True, "_close": True}
    return {"ok": False, "error": f"unknown op {op!r}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="amb-worker")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    try:
        agent = _load_agent(args.config)
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(json.dumps({"ok": False, "error": f"load failed: {exc}",
                                     "traceback": traceback.format_exc()}) + "\n")
        sys.stdout.flush()
        return 1

    sys.stdout.write(json.dumps({"ok": True, "ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            reply = _handle(agent, msg)
        except Exception as exc:  # noqa: BLE001
            reply = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
        close = reply.pop("_close", False)
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
        if close:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
