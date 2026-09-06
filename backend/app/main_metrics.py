from __future__ import annotations

from threading import Lock
from time import monotonic

_lock = Lock()
_state = {"requests": 0, "errors": 0, "total_latency_ms": 0.0, "by_path": {}}


def record(path: str, status: int, latency_ms: float) -> None:
    with _lock:
        _state["requests"] += 1
        if status >= 500:
            _state["errors"] += 1
        _state["total_latency_ms"] += latency_ms
        item = _state["by_path"].setdefault(path, {"requests": 0, "errors": 0, "total_latency_ms": 0.0})
        item["requests"] += 1
        if status >= 500:
            item["errors"] += 1
        item["total_latency_ms"] += latency_ms


def snapshot() -> dict:
    with _lock:
        total = _state["requests"]
        return {
            "requests": total,
            "errors": _state["errors"],
            "error_rate": round(_state["errors"] / total, 6) if total else 0.0,
            "avg_latency_ms": round(_state["total_latency_ms"] / total, 3) if total else 0.0,
            "by_path": {
                path: {**v, "avg_latency_ms": round(v["total_latency_ms"] / v["requests"], 3)}
                for path, v in _state["by_path"].items()
            },
        }
