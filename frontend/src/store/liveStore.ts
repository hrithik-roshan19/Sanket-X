import { create } from "zustand";
import type { LiveEvent, LiveEventType } from "../api/types";

export type ConnectionStatus = "connecting" | "open" | "closed";

interface LiveState {
  connectionStatus: ConnectionStatus;
  lastEventType: LiveEventType | null;
  lastEventAt: string | null;
  trainingInProgress: boolean;
  lastTrainingError: string | null;
  recent: LiveEvent[];
  setStatus: (s: ConnectionStatus) => void;
  pushEvent: (e: LiveEvent) => void;
}

const MAX_RECENT = 20;

export const useLiveStore = create<LiveState>((set) => ({
  connectionStatus: "connecting",
  lastEventType: null,
  lastEventAt: null,
  trainingInProgress: false,
  lastTrainingError: null,
  recent: [],
  setStatus: (connectionStatus) => set({ connectionStatus }),
  pushEvent: (e) =>
    set((prev) => {
      const next: Partial<LiveState> = {
        lastEventType: e.event,
        lastEventAt: e.timestamp,
        recent: [e, ...prev.recent].slice(0, MAX_RECENT),
      };
      if (e.event === "training_started") {
        next.trainingInProgress = true;
        next.lastTrainingError = null;
      } else if (e.event === "training_complete") {
        next.trainingInProgress = false;
      } else if (e.event === "training_failed") {
        next.trainingInProgress = false;
        next.lastTrainingError = String(e.payload.error ?? "training failed");
      } else if (e.event === "connected") {
        next.trainingInProgress = Boolean(e.payload.training_in_progress);
      }
      return next;
    }),
}));
