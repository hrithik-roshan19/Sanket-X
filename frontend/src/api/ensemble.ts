import { apiGet } from "./client";
import type { EnsembleDivergenceResponse } from "./types";

export const fetchEnsembleDivergence = (regionId?: string | null) =>
  apiGet<EnsembleDivergenceResponse>(
    regionId ? `/api/ensemble?region_id=${encodeURIComponent(regionId)}` : "/api/ensemble",
  );
