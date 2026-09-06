import { apiGet } from "./client";
import type { AlertsResponse, RiskBand } from "./types";

export const fetchAlerts = (limit = 25, riskBand?: RiskBand) => {
  const q = new URLSearchParams({ limit: String(limit) });
  if (riskBand) q.set("risk_band", riskBand);
  return apiGet<AlertsResponse>(`/api/alerts?${q}`);
};
