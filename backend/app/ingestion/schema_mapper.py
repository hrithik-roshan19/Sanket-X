from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from app.ingestion.canonical_schema import (
    CanonicalVariable,
    ValueType,
    VARIABLE_PLAUSIBLE_RANGE,
)


V = CanonicalVariable

VARIABLE_SYNONYMS: dict[CanonicalVariable, set[str]] = {
    V.RAINFALL_MM: {
        "rain", "rainfall", "precip", "precipitation", "precipitation mm", "apcp",
        "prectotcorr", "prectot", "total precipitation", "rain mm", "rainfall mm",
        "accumulated precipitation", "precip mm",
    },
    V.TEMPERATURE_C: {
        "temp", "temperature", "t2m", "tmp", "air temp", "air temperature", "temp avg",
        "temp c", "temperature c", "2m temperature", "temperature 2m", "tair",
        "temp mean", "mean temperature", "avg temperature", "t2m c", "surface temperature",
    },
    V.HUMIDITY_PCT: {
        "humidity", "rh", "relative humidity", "rh2m", "humidity pct", "rel humidity",
        "relative humidity 2m", "humidity percent", "rh2m pct", "relhum",
    },
    V.PRESSURE_HPA: {
        "pressure", "mslp", "slp", "surface pressure", "sea level pressure",
        "mean sea level pressure", "pressure hpa", "pressure msl", "psfc", "prmsl",
        "station pressure", "air pressure", "mslp hpa", "pres msl", "surface air pressure",
    },
    V.ATMOSPHERIC_MOISTURE_KGM2: {
        "pwat", "precipitable water", "tcwv", "total column water vapour",
        "total column water vapor", "total column integrated water vapour",
        "atmospheric moisture", "column water vapour", "pwat kgm2", "precipitable water kgm2",
    },
    V.SOIL_MOISTURE_PCT: {
        "soil moisture", "soilw", "volumetric soil moisture", "soil water",
        "soil moisture content", "soil moisture pct", "soilw vol", "soil moisture vol",
        "soil moisture 0 to 7cm", "volumetric soil water",
    },
    V.WIND_SPEED_MS: {
        "wind speed", "wind speed ms", "ws10m", "wspd", "windspeed", "sfcwind",
        "wind spd", "10m wind speed", "wind velocity", "wspd10m", "wind speed kmh",
        "wind gust", "gust", "wind speed 10m",
    },
    V.WIND_DIRECTION_DEG: {
        "wind direction", "wind dir", "wdir", "wind direction deg", "winddir",
        "wind from direction", "wdir10m", "wind bearing", "wind direction 10m",
    },
}

U_WIND_TOKENS = {"u wind", "u10", "ugrd", "uwnd", "eastward wind", "u component wind"}
V_WIND_TOKENS = {"v wind", "v10", "vgrd", "vwnd", "northward wind", "v component wind"}

VALUE_TYPE_SYNONYMS: dict[ValueType, set[str]] = {
    ValueType.FORECAST: {
        "forecast", "fcst", "predicted", "prediction", "forecast value", "nwp forecast",
        "model forecast", "operational forecast", "deterministic forecast",
    },
    ValueType.OBSERVED: {
        "observed", "obs", "actual", "actual value", "reanalysis", "era5 reanalysis",
        "verification", "verifying analysis", "measured", "ground truth",
        "reanalysis observed", "merra", "merra2", "era5", "nasa power merra2",
    },
}

DIMENSION_SYNONYMS: dict[str, set[str]] = {
    "region": {"region", "state", "state ut", "province", "subdivision", "zone", "city",
               "location", "station name", "district", "station", "name"},
    "lat": {"lat", "latitude"},
    "lon": {"lon", "long", "longitude"},
    "valid_date": {"valid date", "valid time", "date", "time", "datetime", "timestamp",
                   "valid", "fcst valid", "observation date", "obs date"},
    "init_date": {"init date", "init time", "initialization time", "initialisation time",
                  "forecast initialization time", "run time", "cycle", "analysis time",
                  "base time", "issued", "init"},
    "lead_time_days": {"lead time", "lead day", "lead", "fhr", "forecast hour",
                       "forecast hours", "lead time days", "lead time hours",
                       "forecast lead", "horizon", "step hours"},
    "ensemble_member_id": {"member", "ensemble member", "ensemble member id", "ens member",
                           "perturbation", "realization"},
}

VARIABLE_NAME_HEADERS = {"forecast variable", "variable", "parameter", "param", "element",
                         "field", "quantity"}
VALUE_COLUMN_HEADERS = {
    "forecast value": ValueType.FORECAST.value,
    "actual value": ValueType.OBSERVED.value,
    "observed value": ValueType.OBSERVED.value,
    "obs value": ValueType.OBSERVED.value,
    "value": None,
}

ROLE_DIMENSION = "dimension"
ROLE_MEASUREMENT = "measurement"
ROLE_VALUE_TYPE = "value_type"
ROLE_VARIABLE_NAME = "variable_name"
ROLE_VALUE = "value"
ROLE_UNMAPPED = "unmapped"

AUTO_CONF = 0.85
AUTO_GAP = 0.15
CONFIRM_FLOOR = 0.40
VALUE_TYPE_OK = 0.70
FUZZY_FLOOR = 82
FUZZY_MIN_LEN = 4
NUMERIC_FRAC_MIN = 0.5

_UNIT_PAREN = re.compile(r"[\(\[\{].*?[\)\]\}]")
_SPLIT = re.compile(r"[^a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def normalize_header(raw: str) -> str:
    s = _CAMEL.sub(" ", str(raw))
    s = _UNIT_PAREN.sub(" ", s)
    s = _SPLIT.sub(" ", s.lower())
    return " ".join(s.split())


def _tok(norm: str) -> frozenset:
    return frozenset(norm.split())


def fingerprint(headers: Sequence[str]) -> str:
    norm = sorted(normalize_header(h) for h in headers)
    return hashlib.sha256("|".join(norm).encode()).hexdigest()


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = {normalize_header(x) for x in a}, {normalize_header(x) for x in b}
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def _synonym_score(norm: str, synonyms: set[str]) -> float:
    tokens = _tok(norm)
    best = 0.0
    for syn in synonyms:
        st = _tok(syn)
        if not st:
            continue
        if tokens == st:
            return 1.0
        if st <= tokens:
            best = max(best, 0.9)
        elif tokens <= st:
            best = max(best, 0.85)
        elif syn in norm or norm in syn:
            best = max(best, 0.75)
        elif len(syn) >= FUZZY_MIN_LEN and len(norm) >= FUZZY_MIN_LEN:
            fz = fuzz.token_set_ratio(norm, syn)
            if fz >= FUZZY_FLOOR:
                best = max(best, 0.6 * (fz - FUZZY_FLOOR) / (100 - FUZZY_FLOOR))
    return best


def _numeric_fraction(series: pd.Series) -> float:
    nn = series.dropna()
    if nn.empty:
        return 0.0
    return float(pd.to_numeric(nn, errors="coerce").notna().mean())


def _numeric_sample(series: pd.Series, n: int = 500) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(vals) > n:
        vals = vals[np.linspace(0, len(vals) - 1, n).astype(int)]
    return vals


def _range_score(sample: np.ndarray, lo: float, hi: float) -> float:
    if sample.size == 0:
        return 0.0
    return float(np.mean((sample >= lo) & (sample <= hi)))


def _detect_unit_conversion(norm: str, variable: CanonicalVariable) -> Optional[str]:
    if variable == V.WIND_SPEED_MS and re.search(r"\bkm ?h\b|kmh|km per h|kph|km hr", norm):
        return "kmh_to_ms"
    if variable == V.TEMPERATURE_C and re.search(r"\bk\b|kelvin", norm):
        return "K_to_C"
    if variable == V.PRESSURE_HPA and re.search(r"\bpa\b", norm) and "hpa" not in norm:
        return "Pa_to_hPa"
    if variable == V.SOIL_MOISTURE_PCT and re.search(r"frac|m3 m3|proportion|vol frac", norm):
        return "frac_to_pct"
    return None


def _sanity_ok(variable: CanonicalVariable, sample: np.ndarray) -> bool:
    if sample.size == 0:
        return True
    lo, hi = VARIABLE_PLAUSIBLE_RANGE[variable]
    if variable in (V.HUMIDITY_PCT, V.SOIL_MOISTURE_PCT):
        frac = np.mean((sample >= 0) & (sample <= 100))
        if variable == V.SOIL_MOISTURE_PCT:
            frac = max(frac, np.mean((sample >= 0) & (sample <= 1)))
        return frac >= 0.9
    if variable == V.PRESSURE_HPA:
        return max(
            np.mean((sample >= lo) & (sample <= hi)),
            np.mean((sample >= 80000) & (sample <= 110000)),
        ) >= 0.8
    if variable == V.TEMPERATURE_C:
        return max(
            np.mean((sample >= lo) & (sample <= hi)),
            np.mean((sample >= 223) & (sample <= 333)),
        ) >= 0.6
    return np.mean((sample >= lo) & (sample <= hi)) >= 0.6


@dataclass
class ColumnProposal:
    source_column: str
    normalized: str
    role: str
    sample_values: list
    suggested_variable: Optional[str] = None
    suggested_value_type: Optional[str] = None
    confidence: float = 0.0
    ambiguity_gap: float = 0.0
    method: Optional[str] = None
    unit_conversion: Optional[str] = None
    decision: str = ROLE_UNMAPPED
    alternatives: list = field(default_factory=list)


@dataclass
class MappingResult:
    proposals: list
    layout: str
    auto_accepted: bool
    profile_match: str
    profile_id: Optional[int] = None
    fingerprint: str = ""
    value_type_column: Optional[str] = None
    notes: list = field(default_factory=list)

    @property
    def needs_confirmation(self) -> bool:
        return not self.auto_accepted

    def by_role(self, role: str) -> list:
        return [p for p in self.proposals if p.role == role]

    def measurement_map(self) -> dict:
        out = {}
        for p in self.proposals:
            if p.role == ROLE_MEASUREMENT and p.decision in ("auto_accept", "confirmed"):
                out[p.source_column] = {
                    "variable": p.suggested_variable,
                    "value_type": p.suggested_value_type,
                    "unit_conversion": p.unit_conversion,
                }
        return out


class SchemaMapper:
    def __init__(self, filename_hint: Optional[str] = None):
        self.filename_hint = (filename_hint or "").lower()


    def _value_type_from_filename(self) -> tuple[Optional[str], float]:
        fn = self.filename_hint
        if re.search(r"reforecast|forecast|gefs|_fcst|nwp", fn):
            return ValueType.FORECAST.value, 0.6
        if re.search(r"observ|era5|reanalys|merra|analysis|_obs|nasa|power", fn):
            return ValueType.OBSERVED.value, 0.6
        return None, 0.0

    def _score_value_type_header(self, norm: str) -> tuple[Optional[str], float]:
        best_vt, best = None, 0.0
        for vt, syns in VALUE_TYPE_SYNONYMS.items():
            s = _synonym_score(norm, syns)
            if s > best:
                best_vt, best = vt.value, s
        return best_vt, best


    def _find_value_type_column(self, df: pd.DataFrame) -> Optional[str]:
        known = set().union(*VALUE_TYPE_SYNONYMS.values())
        for col in df.columns:
            uniq = {normalize_header(v) for v in df[col].dropna().astype(str).unique()[:20]}
            if not uniq or len(uniq) > 6:
                continue
            hits = sum(1 for u in uniq if any(k in u for k in known))
            if hits >= max(1, len(uniq) - 1):
                return col
        return None

    def _find_long_layout(self, df: pd.DataFrame) -> tuple[Optional[str], list]:
        norms = {c: normalize_header(c) for c in df.columns}
        var_col = next((c for c, n in norms.items() if n in VARIABLE_NAME_HEADERS), None)
        val_cols = [c for c, n in norms.items() if n in VALUE_COLUMN_HEADERS]
        if var_col and val_cols:
            vals = {normalize_header(v) for v in df[var_col].dropna().astype(str).unique()[:40]}
            known = set().union(*VARIABLE_SYNONYMS.values())
            if (vals & known) or len(vals) <= 12:
                return var_col, val_cols
        return None, []


    def _classify_column(
        self,
        col: str,
        series: pd.Series,
        *,
        value_type_column: Optional[str],
        long_var_col: Optional[str],
        long_val_cols: list,
    ) -> ColumnProposal:
        norm = normalize_header(col)
        tokens = _tok(norm)
        sample_vals = [
            (None if pd.isna(v) else str(v)[:60])
            for v in series.dropna().astype(str).unique()[:8]
        ]
        p = ColumnProposal(col, norm, ROLE_UNMAPPED, sample_vals)

        if col == value_type_column:
            p.role, p.decision, p.method, p.confidence = ROLE_VALUE_TYPE, "auto_accept", "value_set", 1.0
            return p
        if col == long_var_col:
            p.role, p.decision, p.method, p.confidence = ROLE_VARIABLE_NAME, "auto_accept", "structural", 1.0
            return p
        if col in long_val_cols:
            p.role, p.method, p.confidence = ROLE_VALUE, "structural", 1.0
            vt = VALUE_COLUMN_HEADERS.get(norm)
            if vt is None and value_type_column is None:
                fn_vt, fn_conf = self._value_type_from_filename()
                vt = fn_vt
                p.method = "structural+filename" if fn_vt else "structural"
            p.suggested_value_type = vt
            p.decision = "auto_accept" if (vt is not None or value_type_column is not None) \
                else "needs_confirmation"
            return p

        dim_best, dim_score = None, 0
        for dim, syns in DIMENSION_SYNONYMS.items():
            for syn in syns:
                st = _tok(syn)
                if not st:
                    continue
                if tokens == st:
                    s = 3
                elif len(st) >= 2 and st <= tokens:
                    s = 2
                else:
                    s = 0
                if s > dim_score:
                    dim_best, dim_score = dim, s
        if dim_best is not None:
            p.role, p.suggested_variable = ROLE_DIMENSION, dim_best
            p.decision, p.method, p.confidence = "auto_accept", "synonym", 1.0
            return p

        numeric_frac = _numeric_fraction(series)

        if numeric_frac >= NUMERIC_FRAC_MIN:
            if any(_tok(t) == tokens or _tok(t) <= tokens for t in U_WIND_TOKENS):
                p.role, p.suggested_variable = ROLE_MEASUREMENT, "u_wind"
                p.decision, p.method, p.confidence = "needs_confirmation", "synonym", 0.6
                return p
            if any(_tok(t) == tokens or _tok(t) <= tokens for t in V_WIND_TOKENS):
                p.role, p.suggested_variable = ROLE_MEASUREMENT, "v_wind"
                p.decision, p.method, p.confidence = "needs_confirmation", "synonym", 0.6
                return p

        if numeric_frac < NUMERIC_FRAC_MIN:
            p.role, p.decision, p.method = ROLE_UNMAPPED, "unmapped", "non_numeric"
            return p

        sample = _numeric_sample(series)
        scored = []
        for var, syns in VARIABLE_SYNONYMS.items():
            syn = _synonym_score(norm, syns)
            lo, hi = VARIABLE_PLAUSIBLE_RANGE[var]
            rng = _range_score(sample, lo, hi)
            conf = float(np.clip(0.65 * syn + 0.35 * rng, 0.0, 1.0))
            scored.append((var, conf, syn, rng))
        scored.sort(key=lambda t: t[1], reverse=True)

        top_var, top_conf, top_syn, top_rng = scored[0]
        second = scored[1][1] if len(scored) > 1 else 0.0
        gap = top_conf - second

        p.role = ROLE_MEASUREMENT
        p.suggested_variable = top_var.value
        p.confidence = round(top_conf, 4)
        p.ambiguity_gap = round(gap, 4)
        p.method = "synonym+range" if top_syn and top_rng else ("synonym" if top_syn else "range")
        p.unit_conversion = _detect_unit_conversion(norm, top_var)
        p.alternatives = [
            {"variable": v.value, "confidence": round(c, 4)}
            for v, c, *_ in scored[1:4] if c > 0.05
        ]

        if value_type_column is not None:
            p.suggested_value_type, vt_conf = None, 1.0
        else:
            vt, vt_conf = self._score_value_type_header(norm)
            if vt_conf < VALUE_TYPE_OK:
                fn_vt, fn_conf = self._value_type_from_filename()
                if fn_conf > vt_conf:
                    vt, vt_conf = fn_vt, fn_conf
            p.suggested_value_type = vt

        sane = _sanity_ok(top_var, sample)
        if top_conf < CONFIRM_FLOOR or not sane and top_syn < 0.75:
            p.decision, p.role = "unmapped", ROLE_UNMAPPED
        elif top_conf >= AUTO_CONF and gap >= AUTO_GAP and sane and vt_conf >= VALUE_TYPE_OK:
            p.decision = "auto_accept"
        else:
            p.decision = "needs_confirmation"
        return p


    def map_table(self, df: pd.DataFrame, existing_profiles: Optional[list] = None) -> MappingResult:
        headers = [str(c) for c in df.columns]
        fp = fingerprint(headers)
        notes: list = []

        value_type_column = self._find_value_type_column(df)
        long_var_col, long_val_cols = self._find_long_layout(df)
        layout = "long" if long_var_col else "wide"
        if value_type_column:
            notes.append(f"value_type taken per-row from column '{value_type_column}'")
        if layout == "long":
            notes.append(f"long layout: variable names in '{long_var_col}', "
                         f"values in {long_val_cols}")

        profile_match, profile_id, applied = "none", None, None
        for profile, jac in _sorted_profiles(existing_profiles, headers):
            if profile.fingerprint == fp or jac >= 0.9:
                profile_match, profile_id, applied = "exact", profile.id, profile.confirmed_mapping_json
                notes.append(f"exact/near match to source profile {profile.id} (jaccard {jac:.2f})")
                break
            if jac >= 0.6:
                profile_match, profile_id, applied = "partial", profile.id, profile.confirmed_mapping_json
                notes.append(f"partial match to source profile {profile.id} (jaccard {jac:.2f}); "
                             "differing columns still need confirmation")
                break

        proposals = [
            self._classify_column(
                c, df[c],
                value_type_column=value_type_column,
                long_var_col=long_var_col,
                long_val_cols=long_val_cols,
            )
            for c in df.columns
        ]

        if applied and profile_match == "exact":
            for p in proposals:
                stored = applied.get(p.source_column)
                if stored:
                    p.role = stored.get("role", p.role)
                    p.suggested_variable = stored.get("variable", p.suggested_variable)
                    p.suggested_value_type = stored.get("value_type", p.suggested_value_type)
                    p.unit_conversion = stored.get("unit_conversion", p.unit_conversion)
                    p.decision, p.method = "confirmed", "profile"
        elif applied and profile_match == "partial":
            for p in proposals:
                stored = applied.get(p.source_column)
                if stored and p.decision == "needs_confirmation":
                    p.suggested_variable = stored.get("variable", p.suggested_variable)
                    p.suggested_value_type = stored.get("value_type", p.suggested_value_type)
                    p.method = "profile"

        if profile_match != "exact":
            claims: dict = {}
            for p in proposals:
                if p.role == ROLE_MEASUREMENT and p.decision in ("auto_accept", "needs_confirmation"):
                    claims.setdefault((p.suggested_variable, p.suggested_value_type), []).append(p)
            for key, group in claims.items():
                if len(group) > 1:
                    for p in group:
                        p.decision = "needs_confirmation"
                        p.method = (p.method or "") + "+collision"
                    notes.append(
                        f"{len(group)} columns map to {key[0]} "
                        f"({', '.join(g.source_column for g in group)}) - needs confirmation"
                    )

        has_usable_measurement = any(
            p.role in (ROLE_MEASUREMENT, ROLE_VALUE) and p.decision in ("auto_accept", "confirmed")
            for p in proposals
        )
        auto = has_usable_measurement and all(
            p.decision in ("auto_accept", "confirmed", "unmapped") for p in proposals
        )

        return MappingResult(
            proposals=proposals,
            layout=layout,
            auto_accepted=auto,
            profile_match=profile_match,
            profile_id=profile_id,
            fingerprint=fp,
            value_type_column=value_type_column,
            notes=notes,
        )


def _sorted_profiles(existing_profiles, headers):
    if not existing_profiles:
        return []
    scored = [(p, jaccard(headers, p.header_list_json or [])) for p in existing_profiles]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored
