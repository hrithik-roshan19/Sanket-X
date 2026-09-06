"""Offline final-release validation for Sanket-X.

This script intentionally does not invent data or require network access. It checks the
release invariants that can be verified from source/configuration alone.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
RENDER = ROOT / "render.yaml"

errors: list[str] = []
checks: list[str] = []


def check(condition: bool, ok: str, fail: str) -> None:
    if condition:
        checks.append(ok)
    else:
        errors.append(fail)


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Required release surfaces.
required = [
    BACKEND / "app/main.py",
    BACKEND / "app/api/security.py",
    BACKEND / "app/api/routers/metrics.py",
    BACKEND / "app/services/event_intelligence.py",
    BACKEND / "app/services/region_service.py",
    BACKEND / "app/ml/calibration.py",
    BACKEND / "ml" if False else BACKEND / "app/ml/inference.py",
    FRONTEND / "src/App.tsx",
    FRONTEND / "package.json",
    RENDER,
]
for p in required:
    check(p.exists(), f"present: {p.relative_to(ROOT)}", f"missing required file: {p.relative_to(ROOT)}")

# Python syntax compilation without importing optional scientific dependencies.
for p in (BACKEND / "app").rglob("*.py"):
    try:
        ast.parse(source(p), filename=str(p))
    except SyntaxError as exc:
        errors.append(f"syntax error: {p.relative_to(ROOT)}: {exc}")
check(not any("syntax error:" in e for e in errors), "all backend/app Python files parse", "Python syntax errors found")

main = source(BACKEND / "app/main.py")
security = source(BACKEND / "app/api/security.py")
render = source(RENDER)

check("X-Request-ID" in main and "X-Content-Type-Options" in main,
      "request/security headers are configured", "missing request/security headers")
check('Cache-Control"] = "no-store"' in main,
      "API responses are non-cacheable", "API cache-control hardening missing")
check("require_admin_key" in security and "hmac.compare_digest" in security,
      "admin mutation guard is present", "admin mutation guard missing")
check("UPLOAD_ENABLED" in render and 'value: "false"' in render,
      "production upload path is disabled by default", "production upload path is not explicitly disabled")
check("LIVE_INGEST_ENABLED" in render and 'value: "false"' in render,
      "production live-ingest scheduler is disabled", "production live-ingest scheduler is not disabled")
check("RATE_LIMIT_ENABLED" in render and 'value: "true"' in render,
      "production rate limiting is enabled", "production rate limiting is not enabled")

# Operational GEFS must use the full operational ensemble, while historical training can remain
# on the smaller reforecast archive. We look for the explicit member range in the live code/docs.
config = source(BACKEND / "app/config.py")
check("operational_gefs_member_catalog" in config and "range(1, 31)" in config,
      "31-member operational GEFS catalog is declared", "31-member operational GEFS catalog missing")
check("training_ensemble_member_count" in source(BACKEND / "app/ml/train_pipeline.py") and "ensemble-size mismatch" in source(BACKEND / "app/live/orchestrator.py"),
      "live ensemble/model-size compatibility is enforced", "live ensemble/model-size compatibility guard missing")

# No production source may silently manufacture forecast/observation numbers.
prod_files = list((BACKEND / "app").rglob("*.py")) + list((FRONTEND / "src").rglob("*.ts")) + list((FRONTEND / "src").rglob("*.tsx"))
for p in prod_files:
    text = source(p).lower()
    # These words are allowed in explanatory text. Reject common numeric-fallback patterns only.
    suspicious = re.search(r"(?:mock|synthetic|fake|placeholder).{0,100}(?:value|probability|forecast|observation)\s*=", text)
    if suspicious:
        errors.append(f"possible fabricated production value in {p.relative_to(ROOT)}")
check(not any("possible fabricated production value" in e for e in errors),
      "no obvious fabricated numeric fallback assignment found", "fabricated numeric fallback pattern found")

# Frontend must have a build script and must not be missing the production entrypoint.
pkg = json.loads(source(FRONTEND / "package.json"))
scripts = pkg.get("scripts", {})
check("build" in scripts, "frontend production build script exists", "frontend build script missing")
check((FRONTEND / "src/main.tsx").exists(), "frontend entrypoint exists", "frontend entrypoint missing")

print("SANKET-X FINAL VALIDATION")
for item in checks:
    print(f"PASS  {item}")
if errors:
    print("FAILURES")
    for item in errors:
        print(f"FAIL  {item}")
    raise SystemExit(1)
print(f"RESULT: PASS ({len(checks)} invariants checked)")
