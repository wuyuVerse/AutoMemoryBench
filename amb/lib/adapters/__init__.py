"""amb.lib.adapters — out-of-process system isolation.

A memory/agent system runs in its OWN interpreter (its per-system venv) and talks
to the host over a tiny JSON-lines RPC. The host (amb's clean env) never imports
a provider SDK; it drives a `WorkerAgent` proxy that the existing runner treats as
an ordinary BlackBoxAgent. This replaces the in-process `sys.path.insert(venv)`
boundary with a real process boundary.
"""
