export interface MappingProposal {
  source_column: string;
  normalized: string;
  role: "measurement" | "dimension" | "value_type" | "variable_name" | "value" | "unmapped";
  sample_values: string[];
  suggested_variable: string | null;
  suggested_value_type: string | null;
  confidence: number;
  ambiguity_gap: number;
  method: string | null;
  unit_conversion: string | null;
  decision: "auto_accept" | "needs_confirmation" | "confirmed" | "unmapped";
  alternatives: { variable: string; confidence: number }[];
}

export interface UploadResponse {
  batch_id: string;
  status: "pending_confirmation" | "training_started" | "failed";
  detected_format: string | null;
  layout: string | null;
  row_count_raw: number;
  row_count_ingested: number;
  skipped_rows: number;
  canonical_variables_found: string[];
  region_resolution_rate: number;
  source_profile_match: string;
  mapping_proposals: MappingProposal[];
  notes: string[];
}

export interface ConfirmMappingItem {
  source_column: string;
  variable?: string | null;
  value_type?: string | null;
  role?: string | null;
  unit_conversion?: string | null;
}

export interface RegionSummary {
  region_id: string;
  region_name: string | null;
  bust_probability: number | null;
  risk_band: RiskBand | null;
  confidence: number | null;
  dominant_variable: string | null;
  data_available: boolean;
}

export type RiskBand = "low" | "medium" | "high";

export interface RegionsResponse {
  lead_time_days: number;
  model_trained: boolean;
  last_trained_at: string | null;
  current_run_id: string | null;
  init_date: string | null;
  valid_date: string | null;
  risk_band_definitions: Record<string, string>;
  regions: RegionSummary[];
  message: string | null;

  available_lead_days: number[];
}

export interface AllRegionsResponse {
  model_trained: boolean;
  last_trained_at: string | null;
  current_run_id: string | null;
  init_date: string | null;
  risk_band_definitions: Record<string, string>;
  available_lead_days: number[];
  days: RegionsResponse[];
  message: string | null;
}

export interface VariablePoint {
  lead_time_days: number;
  valid_date: string | null;
  predicted_value: number | null;
  observed_value: number | null;

  observed_status: "final" | "provisional" | null;
  predicted_error: number | null;
  confidence: number | null;
  ensemble_spread: number | null;
  ensemble_member_count: number | null;
}

export interface VariableSeries {
  variable: string;
  available: boolean;
  unit: string | null;
  bust_threshold: number | null;
  model_mae: number | null;
  model_rmse: number | null;
  model_r2: number | null;
  metrics_split: string | null;
  points: VariablePoint[];
}

export interface BustProbabilityPoint {
  lead_time_days: number;
  valid_date: string | null;
  bust_probability: number;
  risk_band: RiskBand;
  dominant_variable: string | null;
}

export interface TopFactor {
  feature: string;
  importance: number;
  method: string;
}

export interface RegionDetailResponse {
  region_id: string;
  region_name: string | null;
  model_trained: boolean;
  current_run_id: string | null;
  init_date: string | null;
  variables: VariableSeries[];
  bust_probability_curve: BustProbabilityPoint[];
  top_factors: TopFactor[];
  top_factors_method: string | null;
  analog_cases: unknown[];
  message: string | null;
}

export interface Alert {
  alert_id: string;
  region_id: string;
  region_name: string | null;
  lead_time_days: number;
  valid_date: string | null;
  bust_probability: number;
  risk_band: RiskBand;
  dominant_variable: string | null;
  created_at: string;
  training_run_id: string | null;
}

export interface AlertsResponse {
  generated_at: string;
  model_trained: boolean;
  risk_band_definitions: Record<string, string>;
  alerts: Alert[];
  message: string | null;
}

export interface ModelStatusResponse {
  model_trained: boolean;
  current_run_id: string | null;
  last_trained_at: string | null;
  training_in_progress: boolean;
  last_training_error: string | null;
  data_volume: {
    total_rows?: number;
    batches?: number;
    regions?: number;
    valid_date_min?: string | null;
    valid_date_max?: string | null;
    forecast_cycles?: number;
    init_date_min?: string | null;
    init_date_max?: string | null;
    by_variable?: Record<string, Record<string, number>>;
    grain_counts?: Record<string, number>;

    unavailable_reason?: string | null;
  };

  training_data?: {
    cycles?: number | null;
    train_cycles?: number | null;
    val_cycles?: number | null;
    held_out_cycles?: number | null;
    canonical_rows?: number | null;
    paired_rows?: number | null;
    first_train_date?: string | null;
  };

  baselines?: {
    run_id?: string;
    test_events?: number;
    test_cycles?: number;
    bust_rate?: number;
    lead_bust_correlation?: { train?: number | null; test?: number | null };
    models?: {
      name: string; brier: number | null; bss: number | null;
      roc_auc: number | null; f1: number | null; is_model?: boolean;
    }[];
  };
  modelled_variables: string[];
  skipped_variables: Record<string, string>;
  validation_metrics: {
    regressors?: Record<string, {
      mae: number | null; rmse: number | null; r2: number | null;
      baseline_mae: number | null; split: string | null; n: number | null;
    }>;
    classifier?: Record<string, number | string | null>;
  };
  thresholds: {
    bust_threshold?: Record<string, number>;
    p90_error?: Record<string, number>;
    risk_band_cuts?: { medium: number; high: number };
    threshold_percentile?: number;
  };
  explanation_method: string | null;
  websocket_clients: number;
  message: string | null;
}

export type LiveEventType =
  | "connected"
  | "upload_received"
  | "mapping_pending"
  | "training_started"
  | "training_complete"
  | "training_failed"
  | "new_alert";

export interface LiveEvent {
  event: LiveEventType;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface IngestRunInfo {
  target: string | null;
  status: "running" | "complete" | "skipped" | "failed";
  trigger: string;
  rows_ingested: number;
  started_at: string | null;
  finished_at: string | null;
  detail: string | null;
  error: string | null;
}

export interface IngestStatus {
  enabled: boolean;
  cycles_watched: string[];
  members: string[];
  last_forecast: IngestRunInfo | null;
  last_observations_provisional: IngestRunInfo | null;
  last_observations_final: IngestRunInfo | null;
  last_cycle_ingested: string | null;
  latest_forecast_init_date: string | null;
  scheduler: {
    running: boolean;
    enabled: boolean;
    tick_seconds: number;
    ticks_completed: number;
    last_tick: string | null;
    next_tick: string | null;
  };
}

export interface ReplayCycleSummary {
  init_date: string;
  lead_days: number[];
  n_regions: number;
  peak_bust_probability: number | null;
  peak_lead_day: number | null;
  peak_region_id: string | null;
  peak_region_name: string | null;
  n_high_regions_peak: number;
  verified: boolean;
  verified_lead_days: number;
  peak_region_abs_error: number | null;
  medium_range_growth: number;
}

export interface ReplayRegionStep {
  region_id: string;
  region_name: string | null;
  bust_probability: number;
  risk_band: RiskBand;
  confidence: number | null;
  dominant_variable: string | null;
}

export interface ReplayLeadStep {
  lead_time_days: number;
  valid_date: string | null;
  regions: ReplayRegionStep[];
  n_high: number;
  n_medium: number;
  mean_bust_probability: number | null;
  narration: string;
}

export interface ReplayFocusPoint {
  lead_time_days: number;
  valid_date: string | null;
  predicted_value: number | null;
  observed_value: number | null;
  observed_status: "final" | "provisional" | null;
  ensemble_spread: number | null;
}

export interface ReplayFocusSeries {
  region_id: string;
  region_name: string | null;
  variable: string;
  unit: string | null;
  bust_threshold: number | null;
  points: ReplayFocusPoint[];
}

export interface ReplayResponse {
  model_trained: boolean;
  current_run_id: string | null;
  init_date: string | null;
  available_cycles: ReplayCycleSummary[];
  steps: ReplayLeadStep[];
  focus: ReplayFocusSeries | null;
  focus_options: ReplayFocusSeries[];
  risk_band_definitions: Record<string, string>;
  summary_narration: string | null;
  message: string | null;
}

export interface EnsemblePoint {
  lead_time_days: number;
  valid_date: string | null;
  value: number | null;
  observed_status: "final" | "provisional" | null;
}

export interface EnsembleMemberTrace {
  member_id: string;
  is_control: boolean;
  points: EnsemblePoint[];
}

export interface NationalRiskPoint {
  lead_time_days: number;
  valid_date: string | null;
  mean_bust_probability: number;
  min_bust_probability: number;
  max_bust_probability: number;
  n_regions: number;
  n_high_regions: number;
}

export interface CalibrationBin {
  bin_lo: number;
  bin_hi: number;
  predicted_mean: number;
  observed_rate: number;
  n: number;
}

export interface ModelSkill {
  split: string | null;
  n: number;
  roc_auc: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  brier: number | null;
  bust_rate: number | null;
  calibration: CalibrationBin[];
  note: string | null;
}

export interface EnsembleDivergenceResponse {
  model_trained: boolean;
  current_run_id: string | null;
  init_date: string | null;
  region_id: string | null;
  region_name: string | null;
  variable: string | null;
  unit: string | null;
  members: EnsembleMemberTrace[];
  ensemble_mean: EnsemblePoint[];
  observed: EnsemblePoint[];
  crossover_lead: number | null;
  peak_bust_probability: number | null;
  mean_bust_probability: number | null;
  prior_mean_bust_probability: number | null;
  prior_init_date: string | null;

  prior_note: string | null;
  n_high_regions: number;
  n_scored_regions: number;
  high_by_lead: number | null;

  spread_growth: number | null;

  subject_reason: string | null;
  source: string | null;
  member_count: number;

  national: NationalRiskPoint[];
  skill: ModelSkill | null;
  national_note: string | null;
  eyebrow: string | null;
  headline_note: string | null;
  message: string | null;
}

export interface IngestRunRow {
  id: number;
  kind: string;
  target: string | null;
  status: "complete" | "skipped" | "failed" | "running" | string;
  trigger: string;
  rows_ingested: number;
  started_at: string | null;
  finished_at: string | null;
  seconds: number | null;
  detail: string | null;
  error: string | null;
}
