import type { TopFactor } from "../../api/types";

export function ShapFactorsList({ factors, method }: { factors: TopFactor[]; method: string | null }) {
  if (!factors.length) return <p className="muted">No explanation available for this region yet.</p>;
  const max = Math.max(...factors.map((f) => f.importance), 1e-9);

  return (
    <div>
      <ul className="factors">
        {factors.map((f) => (
          <li key={f.feature}>
            <span className="factors__name">{f.feature}</span>
            <span className="factors__bar" aria-hidden="true">
              <i style={{ width: `${(f.importance / max) * 100}%` }} />
            </span>
            <span className="factors__value">{f.importance.toFixed(4)}</span>
          </li>
        ))}
      </ul>
      <p className="muted small">
        {method === "shap"
          ? "How much each input pushed this prediction, averaged over the validation split (SHAP)."
          : method === "feature_importance_fallback"
            ? "SHAP unavailable — showing the model's own feature-importance ranking instead, " +
              "which reflects what it relies on overall rather than for this region."
            : "Attribution method unknown."}
      </p>
    </div>
  );
}
