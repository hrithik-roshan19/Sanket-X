import { apiGet } from "./client";
import type { ReplayCycleSummary, ReplayResponse } from "./types";

export const fetchReplayCycles = () => apiGet<ReplayCycleSummary[]>("/api/replay/cycles");

export const fetchReplay = (initDate?: string) =>
  apiGet<ReplayResponse>(
    initDate ? `/api/replay?init_date=${encodeURIComponent(initDate)}` : "/api/replay",
  );
