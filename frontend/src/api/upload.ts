import { apiPostFile, apiPostJson } from "./client";
import type { ConfirmMappingItem, UploadResponse } from "./types";

export const uploadFile = (file: File) => apiPostFile<UploadResponse>("/api/upload", file);

export const confirmMapping = (batchId: string, mappings: ConfirmMappingItem[]) =>
  apiPostJson<UploadResponse>(`/api/upload/${batchId}/confirm-mapping`, { mappings });
