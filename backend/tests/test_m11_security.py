from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_security_headers_and_request_id():
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/api/health", headers={"X-Request-ID": "test-m11"})
        assert r.status_code == 200
        assert r.headers["X-Request-ID"] == "test-m11"
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["Referrer-Policy"] == "no-referrer"


def test_admin_key_dependency_when_configured(monkeypatch):
    from app.api.security import require_admin_key
    monkeypatch.setattr("app.api.security.settings.admin_api_key", "secret")
    with pytest.raises(Exception) as missing:
        require_admin_key(None)
    assert getattr(missing.value, "status_code", None) == 401
    with pytest.raises(Exception) as bad:
        require_admin_key("wrong")
    assert getattr(bad.value, "status_code", None) == 403
    require_admin_key("secret")
    monkeypatch.setattr("app.api.security.settings.admin_api_key", "")


def test_metrics_endpoint_shape():
    from app.main import app
    with TestClient(app) as client:
        client.get("/api/health")
        body = client.get("/api/metrics").json()
        assert body["requests"] >= 2
        assert "error_rate" in body
        assert "avg_latency_ms" in body
        assert "/api/health" in body["by_path"]
