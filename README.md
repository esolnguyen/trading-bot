# Src-First Architecture

This directory is the source root and runtime package for the new trading bot application.

## Rules

- All new bot code lives under `src/`
- `LLM_trader/` is reference-only and must not be imported by the new runtime
- `binance_mcp_server/` stays at an integration boundary unless a minimal safety patch is required
- Legacy code may be copied or adapted into `src/`, but it must be reshaped to match the new contracts and package boundaries

## Package Layout

- `src/app.py`
  Application entrypoint
- `src/bootstrap/`
  Dependency wiring and startup composition
- `src/core/`
  Cross-cutting concerns such as config
- `src/domain/`
  Pure domain models and business rules
- `src/services/`
  Use cases and orchestration logic
- `src/infrastructure/`
  External system adapters such as Binance, AI, Chroma, and data sources
- `src/interfaces/`
  Output and delivery adapters such as console/log notifications
- `src/shared/`
  Small generic helpers with no domain ownership

## Design Intent

- Keep domain objects isolated from exchange, model, and storage SDKs
- Keep orchestration in services, not in low-level adapters
- Prefer narrow adapters around external systems instead of leaking third-party clients through the app
- Move legacy code in small pieces, then rename and simplify it to match this structure
