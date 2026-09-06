export const CHART = {

  forecast: "#2b4eff",

  observed: "#0b1220",

  error: "#f5254a",

  member: "#a9b6d6",

  marker: "#f08700",

  low: "#00a882",
  medium: "#f08700",
  high: "#f5254a",

  grid: "#e5e9f0",
  axis: "#7b8798",
} as const;

export const BAND_LABEL: Record<string, string> = {
  low: "low",
  medium: "watch",
  high: "bust",
};

export const bandLabel = (band: string | null | undefined): string =>
  band ? BAND_LABEL[band] ?? band : "no data";
