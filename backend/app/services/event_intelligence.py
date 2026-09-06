from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


EVENT_RULES = {
    "heavy_rain": {"rainfall_mm": 25.0},
    "extreme_rain": {"rainfall_mm": 64.5},
    "heatwave_like": {"temperature_c": 40.0},
    "high_wind": {"wind_speed_ms": 17.2},
    "high_humidity": {"humidity_pct": 85.0},
    "pressure_anomaly": {"pressure_hpa": 1005.0},
}

IMPACT_ACTIONS = {
    "heavy_rain": (
        ["urban flooding", "transport disruption", "waterlogging"],
        ["verify drainage/flood-prone locations", "review transport contingency plans", "avoid treating rainfall as guaranteed impact"],
    ),
    "extreme_rain": (
        ["flash flooding", "landslide exposure", "transport disruption"],
        ["escalate rainfall verification", "check flood and landslide-prone zones", "prepare contingency communications"],
    ),
    "heatwave_like": (
        ["heat stress", "power demand", "outdoor-work disruption"],
        ["review heat-health advisories", "check vulnerable population preparedness", "monitor temperature observations"],
    ),
    "high_wind": (
        ["transport disruption", "power infrastructure", "outdoor operations"],
        ["review wind-sensitive operations", "check infrastructure exposure", "verify observations before escalation"],
    ),
    "high_humidity": (
        ["heat discomfort", "fog/visibility in some setups"],
        ["monitor temperature-humidity combination", "check visibility observations where relevant"],
    ),
    "pressure_anomaly": (
        ["synoptic transition", "forecast uncertainty"],
        ["review nearby pressure tendencies", "use this as context rather than a standalone warning"],
    ),
}


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def classify_event(event: pd.Series | dict) -> dict:
    row = event.to_dict() if isinstance(event, pd.Series) else dict(event)
    triggered = []
    evidence = {}
    for name, rule in EVENT_RULES.items():
        var, threshold = next(iter(rule.items()))
        value = row.get(var)
        if not _finite(value):
            continue
        value = float(value)
        # Pressure is a contextual heuristic: unusually low pressure is the signal.
        hit = value <= threshold if name == "pressure_anomaly" else value >= threshold
        if hit:
            triggered.append(name)
            evidence[name] = {"variable": var, "value": value, "threshold": threshold}
    return {"event_types": triggered, "evidence": evidence}


def build_event_intelligence(event: pd.Series | dict) -> dict:
    result = classify_event(event)
    impacts: list[str] = []
    actions: list[str] = []
    for event_type in result["event_types"]:
        i, a = IMPACT_ACTIONS.get(event_type, ([], []))
        impacts.extend(i)
        actions.extend(a)
    # Stable de-duplication keeps API output deterministic.
    result["potential_impacts"] = list(dict.fromkeys(impacts))
    result["recommended_actions"] = list(dict.fromkeys(actions))
    return result


def add_event_intelligence(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    if out.empty:
        out["event_types"] = pd.Series(dtype=object)
        out["potential_impacts"] = pd.Series(dtype=object)
        out["recommended_actions"] = pd.Series(dtype=object)
        return out
    details = [build_event_intelligence(r) for _, r in out.iterrows()]
    out["event_types"] = [d["event_types"] for d in details]
    out["potential_impacts"] = [d["potential_impacts"] for d in details]
    out["recommended_actions"] = [d["recommended_actions"] for d in details]
    return out


def _numeric_vector(row: pd.Series, columns: Iterable[str]) -> np.ndarray:
    vals = []
    for c in columns:
        v = row.get(c, np.nan)
        vals.append(float(v) if _finite(v) else np.nan)
    return np.asarray(vals, dtype=float)


def find_analogs(current: pd.Series | dict, history: pd.DataFrame, feature_columns: list[str], k: int = 5) -> list[dict]:
    """Find historical analogs using standardized Euclidean distance.

    Only historical rows are accepted. The current row is never returned, and rows with
    missing comparable features are skipped rather than imputed from future observations.
    """
    if history is None or history.empty or not feature_columns or k <= 0:
        return []
    cur = current if isinstance(current, pd.Series) else pd.Series(current)
    cv = _numeric_vector(cur, feature_columns)
    if not np.isfinite(cv).any():
        return []
    matrix = []
    rows = []
    for idx, r in history.iterrows():
        if "init_date" in history.columns and "init_date" in cur.index:
            if pd.notna(r.get("init_date")) and pd.notna(cur.get("init_date")) and str(r.get("init_date")) == str(cur.get("init_date")):
                continue
        v = _numeric_vector(r, feature_columns)
        mask = np.isfinite(cv) & np.isfinite(v)
        if int(mask.sum()) < max(2, int(np.ceil(len(feature_columns) * 0.5))):
            continue
        matrix.append(v)
        rows.append(r)
    if not matrix:
        return []
    mat = np.vstack(matrix)
    scale = np.nanstd(mat, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    distances = []
    for i, v in enumerate(mat):
        mask = np.isfinite(cv) & np.isfinite(v)
        d = float(np.sqrt(np.mean(((cv[mask] - v[mask]) / scale[mask]) ** 2)))
        distances.append((d, i))
    distances.sort(key=lambda x: x[0])
    result = []
    for d, i in distances[:k]:
        r = rows[i]
        result.append({
            "distance": round(d, 4),
            "init_date": str(r.get("init_date")) if pd.notna(r.get("init_date")) else None,
            "valid_date": str(r.get("valid_date")) if pd.notna(r.get("valid_date")) else None,
            "region_id": str(r.get("region_id")) if pd.notna(r.get("region_id")) else None,
            "lead_time_days": int(r.get("lead_time_days")) if pd.notna(r.get("lead_time_days")) else None,
            "bust_probability": float(r.get("bust_probability")) if _finite(r.get("bust_probability")) else None,
            "risk_band": str(r.get("risk_band")) if pd.notna(r.get("risk_band")) else None,
        })
    return result
