import { useQuery } from "@tanstack/react-query";
import { fetchAlerts } from "../api/alerts";
import { fetchEnsembleDivergence } from "../api/ensemble";
import { fetchIngestRuns, fetchIngestStatus } from "../api/ingest";
import { fetchModelStatus } from "../api/modelStatus";
import { fetchAllRegions, fetchRegionDetail, fetchRegions } from "../api/regions";
import { fetchReplay, fetchReplayCycles } from "../api/replay";
import type { RiskBand } from "../api/types";

const SAFETY_REFETCH_MS = 60_000;

export const useRegions = (leadTimeDays: number) =>
  useQuery({
    queryKey: ["regions", leadTimeDays],
    queryFn: () => fetchRegions(leadTimeDays),
    refetchInterval: SAFETY_REFETCH_MS,
  });

export const useAllRegions = () =>
  useQuery({
    queryKey: ["regions", "all"],
    queryFn: fetchAllRegions,
    staleTime: 5 * 60_000,
    refetchInterval: 5 * 60_000,
  });

export const useEnsembleDivergence = (regionId?: string | null) =>
  useQuery({
    queryKey: ["ensemble", "divergence", regionId ?? "auto"],
    queryFn: () => fetchEnsembleDivergence(regionId),
    staleTime: 5 * 60_000,
    placeholderData: (prev) => prev,
  });

export const useRegionDetail = (regionId: string | null) =>
  useQuery({
    queryKey: ["regions", "detail", regionId],
    queryFn: () => fetchRegionDetail(regionId as string),
    enabled: Boolean(regionId),
  });

export const useAlerts = (limit = 25, riskBand?: RiskBand) =>
  useQuery({
    queryKey: ["alerts", limit, riskBand ?? null],
    queryFn: () => fetchAlerts(limit, riskBand),
    refetchInterval: SAFETY_REFETCH_MS,
  });

export const useModelStatus = () =>
  useQuery({
    queryKey: ["modelStatus"],
    queryFn: fetchModelStatus,
    refetchInterval: SAFETY_REFETCH_MS,
  });

export const useReplayCycles = (enabled: boolean) =>
  useQuery({
    queryKey: ["replay", "cycles"],
    queryFn: fetchReplayCycles,
    enabled,
    staleTime: Infinity,
  });

export const useReplay = (initDate: string | undefined, enabled: boolean) =>
  useQuery({
    queryKey: ["replay", initDate ?? "default"],
    queryFn: () => fetchReplay(initDate),
    enabled,
    staleTime: Infinity,
  });

export const useIngestStatus = () =>
  useQuery({
    queryKey: ["ingestStatus"],
    queryFn: fetchIngestStatus,
    refetchInterval: 30_000,

    retry: false,
  });

export const useIngestRuns = (limit = 25) =>
  useQuery({
    queryKey: ["ingestRuns", limit],
    queryFn: () => fetchIngestRuns(limit),
    refetchInterval: 60_000,
    retry: false,
  });
