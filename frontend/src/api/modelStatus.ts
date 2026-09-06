import { apiGet } from "./client";
import type { ModelStatusResponse } from "./types";

export const fetchModelStatus = () => apiGet<ModelStatusResponse>("/api/model/status");
