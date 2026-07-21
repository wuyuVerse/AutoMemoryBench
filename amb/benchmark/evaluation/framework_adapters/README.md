# Agent Framework Adapter Layer

This directory contains adapters that let AutoMemoryBench evaluate arbitrary agent
frameworks behind a common trace and scoring contract.

Adapters in this layer should:

- translate a framework run into AutoMemoryBench framework traces,
- keep framework-specific optional dependencies isolated,
- avoid importing memory-provider SDKs directly,
- expose clear dependency errors before launching long benchmark runs.

Framework adapters are separate from `amb.clients`, which is only for
memory-system backends. The legacy `amst_real_clients` import path remains a
compatibility shim for existing configs and historical artifacts.
