"""Generate the frontend's region-code mirror from the backend's single source of truth.

The dashboard joins API data (keyed by ISO 3166-2:IN `region_id`) onto the vendored
topojson (keyed by the 2011 census `st_code`). That translation lives in
app/utils/india_state_codes.py; this script emits a TypeScript copy so the two can never
drift by hand-editing. Re-run it if the state table ever changes.

    python scripts/gen_frontend_region_codes.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.utils.india_state_codes import STATES  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "frontend" / "src" / "api" / "regionCodes.ts"

rows = "\n".join(
    f'  {{ regionId: "{s.region_id}", regionName: "{s.region_name}", stCode: "{s.st_code}" }},'
    for s in sorted(STATES, key=lambda x: x.region_id)
)

OUT.write_text(f'''// GENERATED FILE - do not edit by hand.
// Source: backend/app/utils/india_state_codes.py
// Regenerate: python backend/scripts/gen_frontend_region_codes.py
//
// The API keys regions by ISO 3166-2:IN `region_id`; the vendored topojson keys features
// by the 2011 census `st_code`. This table is the only place those two schemes meet.

export interface RegionCode {{
  regionId: string;
  regionName: string;
  stCode: string;
}}

export const REGION_CODES: RegionCode[] = [
{rows}
];

export const REGION_ID_BY_ST_CODE: Record<string, string> = Object.fromEntries(
  REGION_CODES.map((r) => [r.stCode, r.regionId]),
);

export const REGION_NAME_BY_ID: Record<string, string> = Object.fromEntries(
  REGION_CODES.map((r) => [r.regionId, r.regionName]),
);
''')
print(f"wrote {len(STATES)} regions -> {OUT}")
