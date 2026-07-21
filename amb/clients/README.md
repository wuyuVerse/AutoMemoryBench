# AutoMemoryBench Real Clients

Official-code-backed adapters for running external memory systems on
AutoMemoryBench.

## Layout

- `core/`: shared utilities for path resolution, optional virtualenv imports,
  source-tree imports, and environment-based credential lookup.
- `providers/`: stable provider compatibility shims plus implementation
  subpackages. These modules are the only place where official SDKs, official
  source trees, or vendor APIs should be imported.
- `providers/local_sources/`: adapters backed by official source trees or local
  official packages.
- `providers/framework_sdks/`: adapters backed by official framework SDKs.
- `providers/managed_services/`: adapters backed by official hosted services or
  vendor APIs.
- `providers/aliases/`: method-name aliases that preserve benchmark identity.
- `providers/registry.py`: metadata-only registry of method IDs and factory
  paths. Audits use this without importing heavyweight provider dependencies.
- Root modules such as `langmem.py`, `memos.py`, and `openai_memory.py`: stable
  compatibility shims for existing configs and historical artifacts.

## Adapter Rule

Adapters must not reimplement a memory method's core algorithm. They should load
the official package or source tree, translate AutoMemoryBench memory operations
into that API, and surface missing dependencies or credentials explicitly.

Keep secrets in environment variables. Do not commit API keys or generated run
artifacts.
