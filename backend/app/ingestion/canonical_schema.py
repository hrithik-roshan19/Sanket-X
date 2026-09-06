from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CanonicalVariable(str, Enum):

    RAINFALL_MM = "rainfall_mm"
    TEMPERATURE_C = "temperature_c"
    HUMIDITY_PCT = "humidity_pct"
    PRESSURE_HPA = "pressure_hpa"
    ATMOSPHERIC_MOISTURE_KGM2 = "atmospheric_moisture_kgm2"
    SOIL_MOISTURE_PCT = "soil_moisture_pct"
    WIND_SPEED_MS = "wind_speed_ms"
    WIND_DIRECTION_DEG = "wind_direction_deg"


class ValueType(str, Enum):

    FORECAST = "forecast"
    OBSERVED = "observed"


VARIABLE_UNITS: dict[CanonicalVariable, str] = {
    CanonicalVariable.RAINFALL_MM: "mm",
    CanonicalVariable.TEMPERATURE_C: "°C",
    CanonicalVariable.HUMIDITY_PCT: "% RH",
    CanonicalVariable.PRESSURE_HPA: "hPa",
    CanonicalVariable.ATMOSPHERIC_MOISTURE_KGM2: "kg/m² (TCWV)",
    CanonicalVariable.SOIL_MOISTURE_PCT: "% volumetric",
    CanonicalVariable.WIND_SPEED_MS: "m/s",
    CanonicalVariable.WIND_DIRECTION_DEG: "degrees (meteorological, from-direction)",
}

VARIABLE_PLAUSIBLE_RANGE: dict[CanonicalVariable, tuple[float, float]] = {
    CanonicalVariable.RAINFALL_MM: (0.0, 2000.0),
    CanonicalVariable.TEMPERATURE_C: (-50.0, 60.0),
    CanonicalVariable.HUMIDITY_PCT: (0.0, 100.0),
    CanonicalVariable.PRESSURE_HPA: (800.0, 1100.0),
    CanonicalVariable.ATMOSPHERIC_MOISTURE_KGM2: (0.0, 90.0),
    CanonicalVariable.SOIL_MOISTURE_PCT: (0.0, 100.0),
    CanonicalVariable.WIND_SPEED_MS: (0.0, 100.0),
    CanonicalVariable.WIND_DIRECTION_DEG: (0.0, 360.0),
}


class CanonicalRow(BaseModel):

    record_id: str
    upload_batch_id: str
    source_file: str
    source_column: str

    variable: CanonicalVariable
    value_type: ValueType
    value: float

    region_id: Optional[str] = None
    region_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    init_date: Optional[date] = None
    valid_date: date
    lead_time_days: Optional[int] = Field(default=None, ge=1, le=10)

    ensemble_member_id: Optional[str] = None

    mapping_confidence: float = Field(ge=0.0, le=1.0)
    ingested_at: datetime

    grain: str = "native"
    region_resolution_method: Optional[str] = None

    verification_status: Optional[str] = None


CANONICAL_COLUMNS: tuple[str, ...] = tuple(CanonicalRow.model_fields.keys())
