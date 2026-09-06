from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):

    model_config = ConfigDict(protected_namespaces=())


class MappingProposal(Schema):
    source_column: str
    normalized: str
    role: str
    sample_values: list = Field(default_factory=list)
    suggested_variable: Optional[str] = None
    suggested_value_type: Optional[str] = None
    confidence: float = 0.0
    ambiguity_gap: float = 0.0
    method: Optional[str] = None
    unit_conversion: Optional[str] = None
    decision: str = "unmapped"
    alternatives: list = Field(default_factory=list)


class UploadResponse(Schema):
    batch_id: str
    status: str
    detected_format: Optional[str] = None
    layout: Optional[str] = None
    row_count_raw: int = 0
    row_count_ingested: int = 0
    skipped_rows: int = 0
    canonical_variables_found: list = Field(default_factory=list)
    region_resolution_rate: float = 0.0
    source_profile_match: str = "none"
    mapping_proposals: list = Field(default_factory=list)
    notes: list = Field(default_factory=list)


class ConfirmMappingItem(Schema):
    source_column: str
    variable: Optional[str] = None
    value_type: Optional[str] = None
    role: Optional[str] = None
    unit_conversion: Optional[str] = None


class ConfirmMappingRequest(Schema):
    mappings: list


class RegionSummary(Schema):
    region_id: str
    region_name: Optional[str] = None
    bust_probability: Optional[float] = None
    risk_band: Optional[str] = None
    confidence: Optional[float] = None
    reliability_score: Optional[float] = None
    confidence_label: Optional[str] = None
    dominant_variable: Optional[str] = None
    data_available: bool = True


class RegionsResponse(Schema):
    lead_time_days: int
    model_trained: bool
    last_trained_at: Optional[datetime] = None
    current_run_id: Optional[str] = None
    init_date: Optional[date] = None
    valid_date: Optional[date] = None
    risk_band_definitions: dict = Field(default_factory=dict)
    regions: list = Field(default_factory=list)
    message: Optional[str] = None
    available_lead_days: list = Field(default_factory=list)


class AllRegionsResponse(Schema):

    model_trained: bool
    last_trained_at: Optional[datetime] = None
    current_run_id: Optional[str] = None
    init_date: Optional[date] = None
    risk_band_definitions: dict = Field(default_factory=dict)
    available_lead_days: list = Field(default_factory=list)
    days: list = Field(default_factory=list)
    message: Optional[str] = None


class VariablePoint(Schema):
    lead_time_days: int
    valid_date: Optional[date] = None
    predicted_value: Optional[float] = None
    observed_value: Optional[float] = None
    observed_status: Optional[str] = None
    predicted_error: Optional[float] = None
    confidence: Optional[float] = None
    ensemble_spread: Optional[float] = None
    ensemble_member_count: Optional[int] = None


class VariableSeries(Schema):
    variable: str
    available: bool = True
    unit: Optional[str] = None
    bust_threshold: Optional[float] = None
    model_mae: Optional[float] = None
    model_rmse: Optional[float] = None
    model_r2: Optional[float] = None
    metrics_split: Optional[str] = None
    points: list = Field(default_factory=list)


class BustProbabilityPoint(Schema):
    lead_time_days: int
    valid_date: Optional[date] = None
    bust_probability: float
    risk_band: str
    dominant_variable: Optional[str] = None
    reliability_score: Optional[float] = None
    confidence_label: Optional[str] = None
    event_types: list[str] = Field(default_factory=list)
    potential_impacts: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class TopFactor(Schema):
    feature: str
    importance: float
    method: str


class AnalogCase(Schema):
    distance: float
    init_date: Optional[str] = None
    valid_date: Optional[str] = None
    region_id: Optional[str] = None
    lead_time_days: Optional[int] = None
    bust_probability: Optional[float] = None
    risk_band: Optional[str] = None


class RegionDetailResponse(Schema):
    region_id: str
    region_name: Optional[str] = None
    model_trained: bool
    current_run_id: Optional[str] = None
    init_date: Optional[date] = None
    variables: list = Field(default_factory=list)
    bust_probability_curve: list = Field(default_factory=list)
    top_factors: list = Field(default_factory=list)
    top_factors_method: Optional[str] = None
    analog_cases: list[AnalogCase] = Field(default_factory=list)
    event_intelligence: dict = Field(default_factory=dict)
    potential_impacts: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    message: Optional[str] = None


class Alert(Schema):
    alert_id: str
    region_id: str
    region_name: Optional[str] = None
    lead_time_days: int
    valid_date: Optional[date] = None
    bust_probability: float
    risk_band: str
    dominant_variable: Optional[str] = None
    created_at: datetime
    training_run_id: Optional[str] = None


class AlertsResponse(Schema):
    generated_at: datetime
    model_trained: bool
    risk_band_definitions: dict = Field(default_factory=dict)
    alerts: list = Field(default_factory=list)
    message: Optional[str] = None


class ModelStatusResponse(Schema):
    model_trained: bool
    current_run_id: Optional[str] = None
    last_trained_at: Optional[datetime] = None
    training_in_progress: bool = False
    last_training_error: Optional[str] = None
    data_volume: dict = Field(default_factory=dict)
    training_data: dict = Field(default_factory=dict)
    baselines: dict = Field(default_factory=dict)
    modelled_variables: list = Field(default_factory=list)
    skipped_variables: dict = Field(default_factory=dict)
    validation_metrics: dict = Field(default_factory=dict)
    thresholds: dict = Field(default_factory=dict)
    explanation_method: Optional[str] = None
    websocket_clients: int = 0
    message: Optional[str] = None


class ReplayCycleSummary(Schema):
    init_date: date
    lead_days: list = Field(default_factory=list)
    n_regions: int = 0
    peak_bust_probability: Optional[float] = None
    peak_lead_day: Optional[int] = None
    peak_region_id: Optional[str] = None
    peak_region_name: Optional[str] = None
    n_high_regions_peak: int = 0
    verified: bool = False
    verified_lead_days: int = 0
    peak_region_abs_error: Optional[float] = None
    medium_range_growth: float = 0.0


class ReplayRegionStep(Schema):
    region_id: str
    region_name: Optional[str] = None
    bust_probability: float
    risk_band: str
    confidence: Optional[float] = None
    reliability_score: Optional[float] = None
    confidence_label: Optional[str] = None
    dominant_variable: Optional[str] = None


class ReplayLeadStep(Schema):
    lead_time_days: int
    valid_date: Optional[date] = None
    regions: list = Field(default_factory=list)
    n_high: int = 0
    n_medium: int = 0
    mean_bust_probability: Optional[float] = None
    narration: str = ""


class ReplayFocusPoint(Schema):
    lead_time_days: int
    valid_date: Optional[date] = None
    predicted_value: Optional[float] = None
    observed_value: Optional[float] = None
    observed_status: Optional[str] = None
    ensemble_spread: Optional[float] = None


class ReplayFocusSeries(Schema):
    region_id: str
    region_name: Optional[str] = None
    variable: str
    unit: Optional[str] = None
    bust_threshold: Optional[float] = None
    points: list = Field(default_factory=list)


class ReplayResponse(Schema):
    model_trained: bool
    current_run_id: Optional[str] = None
    init_date: Optional[date] = None
    available_cycles: list = Field(default_factory=list)
    steps: list = Field(default_factory=list)
    focus: Optional[ReplayFocusSeries] = None
    focus_options: list = Field(default_factory=list)
    risk_band_definitions: dict = Field(default_factory=dict)
    summary_narration: Optional[str] = None
    message: Optional[str] = None


class EnsemblePoint(Schema):
    lead_time_days: int
    valid_date: Optional[date] = None
    value: Optional[float] = None
    observed_status: Optional[str] = None


class EnsembleMemberTrace(Schema):
    member_id: str
    is_control: bool = False
    points: list = Field(default_factory=list)


class NationalRiskPoint(Schema):
    lead_time_days: int
    valid_date: Optional[date] = None
    mean_bust_probability: float
    min_bust_probability: float
    max_bust_probability: float
    n_regions: int
    n_high_regions: int


class CalibrationBin(Schema):
    bin_lo: float
    bin_hi: float
    predicted_mean: float
    observed_rate: float
    n: int


class ModelSkill(Schema):
    split: Optional[str] = None
    n: int = 0
    roc_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    brier: Optional[float] = None
    bust_rate: Optional[float] = None
    calibration: list = Field(default_factory=list)
    note: Optional[str] = None


class EnsembleDivergenceResponse(Schema):
    model_trained: bool
    current_run_id: Optional[str] = None
    init_date: Optional[date] = None

    region_id: Optional[str] = None
    region_name: Optional[str] = None
    variable: Optional[str] = None
    unit: Optional[str] = None

    members: list = Field(default_factory=list)
    ensemble_mean: list = Field(default_factory=list)
    observed: list = Field(default_factory=list)

    crossover_lead: Optional[int] = None
    peak_bust_probability: Optional[float] = None

    mean_bust_probability: Optional[float] = None
    prior_mean_bust_probability: Optional[float] = None
    prior_init_date: Optional[date] = None
    prior_note: Optional[str] = None

    n_high_regions: int = 0
    n_scored_regions: int = 0
    high_by_lead: Optional[int] = None

    spread_growth: Optional[float] = None
    subject_reason: Optional[str] = None

    source: Optional[str] = None
    member_count: int = 0

    national: list = Field(default_factory=list)
    skill: Optional[ModelSkill] = None
    national_note: Optional[str] = None

    eyebrow: Optional[str] = None
    headline_note: Optional[str] = None
    message: Optional[str] = None
