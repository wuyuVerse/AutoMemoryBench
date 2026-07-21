# Real-Client Provider Layer

This directory contains provider-specific adapters for official memory-system
official SDKs or official source trees. Each adapter is responsible only for
translating AutoMemoryBench operations into the official implementation.

## Layout

- `registry.py`: stable method IDs and public factory paths.
- `<provider>.py`: compatibility shim for one official method or method family.
- `local_sources/`: official source-tree or local official package adapters.
- `framework_sdks/`: official framework SDK adapters.
- `managed_services/`: official hosted-service or vendor API adapters.
- `aliases/`: method-identity aliases that preserve benchmark naming.
- `__init__.py`: package marker only; avoid broad imports that would load heavy
  optional dependencies at import time.

Root modules such as `amb.clients.langmem` are compatibility shims. Do not
put implementation logic there.

## Adding a Provider

1. Add the implementation under the relevant provider-layer subpackage.
2. Add a thin provider shim and root shim if benchmark configs need a stable
   public factory path.
3. Add the method to `registry.py`.
4. Add or update dependency-gate tests and real-client smoke tests.
5. Keep API keys and other credential material environment-driven; never write
   secrets into configs or source.
