from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MISSING_SENTINELS = {"-999", "-999.0", "-9999", "NA", "N/A", "null", "NULL", ""}
_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


@dataclass
class ParsedTable:
    df: pd.DataFrame
    detected_format: str
    layout_hint: Optional[str] = None
    grain: str = "native"
    reshape: Optional[str] = None
    sheet_names: Optional[list] = None
    parse_notes: list = field(default_factory=list)

    @property
    def headers(self) -> list:
        return [str(c) for c in self.df.columns]


class ParseError(ValueError):
    pass


def parse_upload(path: Path | str, original_filename: Optional[str] = None) -> ParsedTable:
    path = Path(path)
    name = (original_filename or path.name).lower()
    ext = Path(name).suffix

    if ext in (".xlsx", ".xls"):
        return _parse_excel(path)
    if ext == ".parquet":
        return _finalise(pd.read_parquet(path), "parquet", [])
    if ext == ".json":
        return _parse_json(path)
    return _parse_delimited(path, ext)


def _parse_delimited(path: Path, ext: str) -> ParsedTable:
    raw = path.read_text(encoding="utf-8", errors="replace")
    notes: list = []

    skiprows, header_note = _nasa_power_header_offset(raw)
    if skiprows:
        notes.append(header_note)

    body = "\n".join(raw.splitlines()[skiprows:])
    sep = _sniff_delimiter(body, ext)
    fmt = {",": "csv", "\t": "tsv"}.get(sep, "csv")

    df = pd.read_csv(
        io.StringIO(body),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=True,
        engine="python",
    )
    df.columns = [str(c).strip() for c in df.columns]
    df = _apply_missing_sentinels(df, notes)

    reshape, grain, layout = None, "native", None
    if _looks_like_power_monthly(df.columns):
        df, notes = _melt_power_monthly(df, notes)
        reshape, grain, layout = "power_monthly_melt", "monthly", "wide"
    elif _looks_like_power_daily(df.columns):
        df, notes = _power_daily_doy_to_date(df, notes)
        reshape, layout = "power_daily_doy", "wide"

    return _finalise(df, fmt, notes, reshape=reshape, grain=grain, layout_hint=layout)


def _sniff_delimiter(body: str, ext: str) -> str:
    if ext == ".tsv":
        return "\t"
    sample = "\n".join(body.splitlines()[:50])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample.splitlines() else ""
        return "\t" if first.count("\t") > first.count(",") else ","


def _nasa_power_header_offset(raw: str) -> tuple[int, str]:
    lines = raw.splitlines()
    for i, line in enumerate(lines[:60]):
        if line.strip().upper() == "-END HEADER-":
            return i + 1, f"skipped {i + 1} NASA POWER header lines (ended at '-END HEADER-')"
    return 0, ""


def _apply_missing_sentinels(df: pd.DataFrame, notes: list) -> pd.DataFrame:
    hits = 0
    for col in df.columns:
        mask = df[col].isin(MISSING_SENTINELS)
        if mask.any():
            hits += int(mask.sum())
            df.loc[mask, col] = np.nan
    if hits:
        notes.append(f"replaced {hits} missing-value sentinels ({sorted(MISSING_SENTINELS)}) with NaN")
    return df


def _looks_like_power_monthly(cols) -> bool:
    upper = {str(c).upper() for c in cols}
    return set(_MONTHS).issubset(upper) and "YEAR" in upper


def _looks_like_power_daily(cols) -> bool:
    upper = {str(c).upper() for c in cols}
    return {"YEAR", "DOY"}.issubset(upper)


def _melt_power_monthly(df: pd.DataFrame, notes: list) -> tuple[pd.DataFrame, list]:
    colmap = {str(c).upper(): c for c in df.columns}
    id_cols = [colmap[c] for c in ("PARAMETER", "YEAR", "LAT", "LON") if c in colmap]
    month_cols = [colmap[m] for m in _MONTHS]
    long = df.melt(
        id_vars=id_cols, value_vars=month_cols, var_name="month_abbr", value_name="value"
    )
    month_num = {m: i + 1 for i, m in enumerate(_MONTHS)}
    long["month_abbr"] = long["month_abbr"].astype(str).str.upper()
    yr = pd.to_numeric(long[colmap["YEAR"]], errors="coerce")
    mo = long["month_abbr"].map(month_num)
    long["valid_date"] = pd.to_datetime(
        dict(year=yr, month=mo, day=1), errors="coerce"
    ).dt.date
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.drop(columns=["month_abbr"])
    notes.append(
        f"melted NASA POWER monthly wide layout ({len(month_cols)} month columns) to long; "
        "valid_date set to first-of-month (grain=monthly)"
    )
    return long, notes


def _power_daily_doy_to_date(df: pd.DataFrame, notes: list) -> tuple[pd.DataFrame, list]:
    colmap = {str(c).upper(): c for c in df.columns}
    yr = pd.to_numeric(df[colmap["YEAR"]], errors="coerce")
    doy = pd.to_numeric(df[colmap["DOY"]], errors="coerce")

    def _mk(y, d):
        if pd.isna(y) or pd.isna(d):
            return pd.NaT
        return date(int(y), 1, 1) + timedelta(days=int(d) - 1)

    df = df.copy()
    df["valid_date"] = [_mk(y, d) for y, d in zip(yr, doy)]
    notes.append("converted NASA POWER YEAR + DOY to valid_date")
    return df, notes


def _parse_excel(path: Path) -> ParsedTable:
    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[0]
    df = xl.parse(sheet, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    notes = [f"read sheet '{sheet}'"]
    if len(xl.sheet_names) > 1:
        notes.append(f"ignored {len(xl.sheet_names) - 1} other sheet(s): {xl.sheet_names[1:]}")
    df = _apply_missing_sentinels(df, notes)
    return _finalise(df, "xlsx", notes, sheet_names=xl.sheet_names)


def _parse_json(path: Path) -> ParsedTable:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if data.get("type") in {"FeatureCollection", "Feature"} or "features" in data:
            raise ParseError("file is GeoJSON, not a tabular dataset")
        for key in ("data", "rows", "records", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ParseError("JSON is not an array of flat records")
    df = pd.json_normalize(data)
    if any("." in c for c in df.columns):
        raise ParseError("JSON records are nested; flatten before uploading")
    df.columns = [str(c).strip() for c in df.columns]
    notes: list = []
    df = _apply_missing_sentinels(df, notes)
    return _finalise(df, "json", notes)


def _finalise(
    df: pd.DataFrame,
    fmt: str,
    notes: list,
    *,
    reshape: Optional[str] = None,
    grain: str = "native",
    layout_hint: Optional[str] = None,
    sheet_names: Optional[list] = None,
) -> ParsedTable:
    df = df.reset_index(drop=True)
    df = df.dropna(axis=1, how="all")
    return ParsedTable(
        df=df,
        detected_format=fmt,
        layout_hint=layout_hint,
        grain=grain,
        reshape=reshape,
        sheet_names=sheet_names,
        parse_notes=notes,
    )
