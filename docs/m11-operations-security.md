# Sanket-X M11 — API, Security, Observability & Deployment

## Scope

M11 hardens the serving layer and connects the operational dashboard to live connection state.
Training remains disabled on the low-memory serving deployment; training belongs on CI/release runners.

## Security controls

- `UPLOAD_ENABLED=false` on the production Render blueprint.
- Optional `ADMIN_API_KEY` protects uploads and live ingestion when configured.
- `X-Admin-Key` is compared with constant-time `hmac.compare_digest`.
- Live mutation endpoints remain blocked when `LIVE_INGEST_ENABLED=false`.
- In-memory rate-limit primitives are available for deployments that explicitly enable `RATE_LIMIT_ENABLED`.
- Security response headers: `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, and `Permissions-Policy`.
- API responses are marked `Cache-Control: no-store` to avoid stale sensitive operational state.
- Request IDs are accepted/generated and returned as `X-Request-ID`.

## Observability

`GET /api/metrics` exposes process-local request count, 5xx count, error rate, average latency, and per-path counters.
It is intentionally dependency-free and is a lightweight demo/deployment diagnostic, not a replacement for a production metrics backend.

## Operational behavior

- WebSocket is an enhancement, not the source of truth.
- Frontend keeps HTTP polling as the safety fallback.
- The serving container does not train.
- CI produces model/data artifacts; deployment serves the last known-good artifact.

## M11 test gate

Before release, run:

```text
python -m compileall backend/app backend/tests
pytest -q
npm ci
npm run build
```

If PyArrow or another declared dependency is unavailable, report an environment failure rather than claiming the suite passed.
