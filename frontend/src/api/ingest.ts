import { apiGet, apiPostJson } from "./client";
import type { IngestRunRow, IngestStatus } from "./types";

export const fetchIngestStatus = () => apiGet<IngestStatus>("/api/ingest/status");

export const triggerCycleRun = () =>
  apiPostJson<{ status: string; target: string; note?: string }>("/api/ingest/run-cycle", {});

export const fetchIngestRuns = (limit = 25) =>
  apiGet<{ runs: IngestRunRow[] }>(`/api/ingest/runs?limit=${limit}`);
