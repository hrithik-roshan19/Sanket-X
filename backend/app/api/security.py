from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from threading import Lock
from uuid import uuid4

from fastapi import Header, HTTPException, Request

from app.config import settings

_lock = Lock()
_buckets: dict[str, deque[float]] = defaultdict(deque)


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or uuid4().hex


def require_admin_key(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")) -> None:
    """Protect state-changing/admin operations when a deployment key is configured.

    Local development intentionally permits calls when ADMIN_API_KEY is unset. Production
    Render config sets UPLOAD_ENABLED=false and LIVE_INGEST_ENABLED=false by default; if
    either is enabled, an admin key should be configured.
    """
    configured = settings.admin_api_key
    if configured and not x_admin_key:
        raise HTTPException(status_code=401, detail="X-Admin-Key is required for this operation")
    if configured and not hmac.compare_digest(x_admin_key or "", configured):
        raise HTTPException(status_code=403, detail="invalid admin key")


def enforce_rate_limit(request: Request, bucket: str) -> None:
    if not settings.rate_limit_enabled:
        return
    now = time.monotonic()
    key = f"{bucket}:{request.client.host if request.client else 'unknown'}"
    window = 60.0
    limit = settings.rate_limit_per_minute
    with _lock:
        q = _buckets[key]
        while q and now - q[0] >= window:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded; try again later")
        q.append(now)


def clear_rate_limits() -> None:
    with _lock:
        _buckets.clear()
