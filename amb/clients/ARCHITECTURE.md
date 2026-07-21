# AutoMemoryBench Real Clients Architecture

`amb.clients` contains official-code-backed adapters for running external
memory systems against AutoMemoryBench. The package is split into three layers:

- `core/`: shared path, import, virtualenv, and credential-environment helpers.
- `providers/`: stable provider shims plus implementation subpackages.
- `providers/local_sources/`: adapters that wrap official source trees or local
  official packages.
- `providers/framework_sdks/`: adapters that wrap official framework SDKs.
- `providers/managed_services/`: adapters that wrap official hosted services or
  vendor APIs.
- `providers/aliases/`: method-identity aliases that map benchmark names to
  official implementations.
- `providers/registry.py`: the method-ID to public factory-path registry used for
  audits and documentation.
- root modules such as `langmem.py` or `openai_memory.py`: compatibility
  entrypoints used by benchmark configs and historical artifacts.

## Import Stability

Benchmark configs intentionally keep stable root factories such as:

```text
amb.clients.langmem:create_client
```

The root module re-exports the provider implementation from:

```text
amb.clients.providers.langmem:create_client
```

This keeps old artifacts reproducible while allowing the source tree to grow into
a clearer open-source layout. New provider implementations should be placed under
the relevant `providers/<layer>/` package; `providers/<method>.py` and root
modules should stay as thin compatibility wrappers unless a breaking config
migration is explicitly planned.

Root provider modules are intentionally small. If a root module grows beyond a
wrapper, move the implementation back under `providers/` and keep only the stable
import shim at the root.

## Provider Registry

Every official adapter should be listed in:

```text
amb.clients.providers.registry:PROVIDER_SPECS
```

The registry is intentionally metadata-only so audits can inspect available
methods without importing heavy optional SDKs or official source trees.

## Official-Code Boundary

Adapters should not reimplement a method's memory algorithm. They should:

- load the official SDK or official source tree,
- translate AMST's `put/search/list/delete` style calls into that API,
- keep credentials environment-based,
- expose traceable errors when official dependencies are missing.

Do not hard-code secret values in this package.
