import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { WS_URL } from "../api/client";
import type { LiveEvent } from "../api/types";
import { useLiveStore } from "../store/liveStore";

export function useLiveSocket() {
  const queryClient = useQueryClient();
  const setStatus = useLiveStore((s) => s.setStatus);
  const pushEvent = useLiveStore((s) => s.pushEvent);
  const retryRef = useRef(0);
  const closedRef = useRef(false);

  useEffect(() => {
    closedRef.current = false;
    let ws: WebSocket | null = null;
    let timer: number | undefined;

    const connect = () => {
      if (closedRef.current) return;
      setStatus("connecting");
      ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        retryRef.current = 0;
        setStatus("open");
      };

      ws.onmessage = (raw) => {
        let event: LiveEvent;
        try {
          event = JSON.parse(raw.data as string);
        } catch {
          return;
        }
        pushEvent(event);

        switch (event.event) {
          case "training_complete":
          case "new_alert":
            queryClient.invalidateQueries({ queryKey: ["regions"] });
            queryClient.invalidateQueries({ queryKey: ["alerts"] });
            queryClient.invalidateQueries({ queryKey: ["modelStatus"] });
            queryClient.invalidateQueries({ queryKey: ["ensemble"] });
            queryClient.invalidateQueries({ queryKey: ["replay"] });
            break;
          case "training_started":
          case "training_failed":
          case "upload_received":
          case "mapping_pending":
          case "connected":
            queryClient.invalidateQueries({ queryKey: ["modelStatus"] });
            break;
        }
      };

      ws.onclose = () => {
        setStatus("closed");
        if (closedRef.current) return;

        const delay = Math.min(30_000, 1000 * 2 ** retryRef.current++);
        timer = window.setTimeout(connect, delay);
      };

      ws.onerror = () => ws?.close();
    };

    connect();
    return () => {
      closedRef.current = true;
      if (timer) window.clearTimeout(timer);
      ws?.close();
    };
  }, [queryClient, setStatus, pushEvent]);
}
