"""Host-side proxy for an out-of-process system worker.

`WorkerAgent` satisfies the runner's BlackBoxAgent contract (reset / observe /
answer_or_act / export_trace) by forwarding each call as JSON-RPC to a child
process running `worker_entry` inside the system's venv. The existing runner
drives it unchanged, so predictions — and therefore scores — are identical to an
in-process run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


class WorkerError(RuntimeError):
    pass


class WorkerAgent:
    def __init__(
        self,
        config_path: str,
        *,
        python_executable: str | None = None,
        cwd: str | Path = REPO_ROOT,
        env: dict[str, str] | None = None,
        startup_timeout: float = 600.0,
    ) -> None:
        self.config_path = str(config_path)
        python = python_executable or sys.executable
        child_env = dict(os.environ if env is None else env)
        # Ensure the worker can import `amb` even from a foreign venv.
        existing = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), existing) if p)
        self._proc = subprocess.Popen(
            [python, "-m", "amb.lib.adapters.worker_entry", "--config", self.config_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(cwd), env=child_env,
        )
        ready = self._read()
        if not ready.get("ready"):
            self.close()
            raise WorkerError(f"worker failed to start: {ready}")

    # ----- BlackBoxAgent surface -----
    def reset(self, case_id: str) -> None:
        self._rpc({"op": "reset", "case_id": case_id})

    def observe(self, observation: dict[str, Any]) -> None:
        self._rpc({"op": "observe", "observation": observation})

    def answer_or_act(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._rpc({"op": "answer_or_act", "request": request})["result"]

    def export_trace(self) -> Any:
        return self._rpc({"op": "export_trace"}).get("result")

    # ----- lifecycle -----
    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"op": "close"}) + "\n")
                proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()

    def __enter__(self) -> "WorkerAgent":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ----- transport -----
    def _read(self) -> dict[str, Any]:
        line = self._proc.stdout.readline() if self._proc.stdout else ""
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise WorkerError(f"worker closed unexpectedly. stderr:\n{err}")
        return json.loads(line)

    def _rpc(self, msg: dict[str, Any]) -> dict[str, Any]:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        reply = self._read()
        if not reply.get("ok"):
            raise WorkerError(f"{msg.get('op')} failed: {reply.get('error')}\n{reply.get('traceback', '')}")
        return reply
