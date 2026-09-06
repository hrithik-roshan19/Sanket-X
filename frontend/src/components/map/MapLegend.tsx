export function MapLegend({ definitions }: { definitions: Record<string, string> }) {
  const bands: { key: string; label: string }[] = [
    { key: "low", label: "low" },
    { key: "medium", label: "watch" },
    { key: "high", label: "bust" },
  ];
  return (
    <div className="legend">
      <span className="legend__title">Bust risk</span>
      {bands.map((b) => (
        <span key={b.key} className="legend__item" title={definitions[b.key] ?? ""}>
          <i className={`swatch swatch--${b.key}`} aria-hidden="true" />
          {b.label}
        </span>
      ))}
      <span className="legend__item legend__item--muted" title="No forecast data for this region">
        <i className="swatch swatch--nodata" aria-hidden="true" />
        No data
      </span>
      {definitions.basis ? <p className="legend__basis">{definitions.basis}</p> : null}
    </div>
  );
}
