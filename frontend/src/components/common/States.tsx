import type { ReactNode } from "react";
import { API_BASE } from "../../api/client";
import { bandLabel } from "../../theme";

export function EmptyState({ title, message, action }: {
  title: string;
  message?: string | null;
  action?: ReactNode;
}) {
  return (
    <div className="state state--empty">
      <strong>{title}</strong>
      {message ? <p>{message}</p> : null}
      {action}
    </div>
  );
}

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return <div className="state state--loading">{label}</div>;
}

export function ErrorState({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="state state--error">
      <strong>Could not reach the API</strong>
      <p>{message}</p>
      <p className="muted">{API_BASE ? `Is the backend running on ${API_BASE}?` : "The API is served from this same origin, so this is a server-side problem rather than a misconfigured address."}</p>
    </div>
  );
}

export function RiskBadge({ band }: { band: string | null }) {
  if (!band) return <span className="badge badge--unknown">unknown</span>;
  return <span className={`badge badge--${band}`}>{bandLabel(band)}</span>;
}

export function StatusBadge({ status, label }: { status: string; label: string }) {
  return <span className={`badge badge--${status}`}>{label}</span>;
}
