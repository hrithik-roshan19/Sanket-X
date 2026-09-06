import { apiGet } from "./client";
import type { AllRegionsResponse, RegionDetailResponse, RegionsResponse } from "./types";

export const fetchRegions = (leadTimeDays: number) =>
  apiGet<RegionsResponse>(`/api/regions?lead_time_days=${leadTimeDays}`);

export const fetchAllRegions = () => apiGet<AllRegionsResponse>("/api/regions/all");

export const fetchRegionDetail = (regionId: string) =>
  apiGet<RegionDetailResponse>(`/api/regions/${encodeURIComponent(regionId)}`);
