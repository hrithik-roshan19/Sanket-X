export function stamp(iso: string | null | undefined): string {
  const d = parseUtc(iso);
  if (!d) return "—";
  return `${istParts(d)} IST`;
}

export function stampShort(iso: string | null | undefined): string {
  const d = parseUtc(iso);
  if (!d) return "—";
  return istParts(d);
}

function parseUtc(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}

function istParts(d: Date): string {
  // Never a hand-added +5:30: a manual offset gives the wrong DATE around midnight.
  const p = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Kolkata",
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hour12: false,
  }).formatToParts(d).reduce<Record<string, string>>((a, x) => ((a[x.type] = x.value), a), {});
  return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}`;
}
