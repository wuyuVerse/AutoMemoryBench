# Integrations

Config-driven wrappers that connect external memory clients to the benchmark
agent interface.

Integrations should stay thin: load a configured factory, adapt calls to the
benchmark runtime, and expose dependency or credential failures explicitly.

