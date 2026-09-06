from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StateRecord:
    region_id: str
    region_name: str
    st_code: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


STATES: tuple[StateRecord, ...] = (
    StateRecord("IN-AN", "Andaman and Nicobar Islands", "35", ("andaman & nicobar islands", "andaman and nicobar")),
    StateRecord("IN-AP", "Andhra Pradesh", "37", ()),
    StateRecord("IN-AR", "Arunachal Pradesh", "12", ()),
    StateRecord("IN-AS", "Assam", "18", ()),
    StateRecord("IN-BR", "Bihar", "10", ()),
    StateRecord("IN-CH", "Chandigarh", "04", ()),
    StateRecord("IN-CT", "Chhattisgarh", "22", ("chattisgarh",)),
    StateRecord("IN-DH", "Dadra and Nagar Haveli and Daman and Diu", "26", ("dadra & nagar haveli and daman & diu", "dadra and nagar haveli", "daman and diu")),
    StateRecord("IN-DL", "Delhi", "07", ("nct of delhi", "national capital territory of delhi", "delhi (nct)", "new delhi")),
    StateRecord("IN-GA", "Goa", "30", ()),
    StateRecord("IN-GJ", "Gujarat", "24", ()),
    StateRecord("IN-HR", "Haryana", "06", ()),
    StateRecord("IN-HP", "Himachal Pradesh", "02", ()),
    StateRecord("IN-JK", "Jammu and Kashmir", "01", ("jammu & kashmir", "j&k")),
    StateRecord("IN-JH", "Jharkhand", "20", ()),
    StateRecord("IN-KA", "Karnataka", "29", ()),
    StateRecord("IN-KL", "Kerala", "32", ()),
    StateRecord("IN-LA", "Ladakh", "38", ()),
    StateRecord("IN-LD", "Lakshadweep", "31", ()),
    StateRecord("IN-MP", "Madhya Pradesh", "23", ()),
    StateRecord("IN-MH", "Maharashtra", "27", ()),
    StateRecord("IN-MN", "Manipur", "14", ()),
    StateRecord("IN-ML", "Meghalaya", "17", ()),
    StateRecord("IN-MZ", "Mizoram", "15", ()),
    StateRecord("IN-NL", "Nagaland", "13", ()),
    StateRecord("IN-OR", "Odisha", "21", ("orissa",)),
    StateRecord("IN-PY", "Puducherry", "34", ("pondicherry",)),
    StateRecord("IN-PB", "Punjab", "03", ()),
    StateRecord("IN-RJ", "Rajasthan", "08", ()),
    StateRecord("IN-SK", "Sikkim", "11", ()),
    StateRecord("IN-TN", "Tamil Nadu", "33", ()),
    StateRecord("IN-TG", "Telangana", "36", ("telengana",)),
    StateRecord("IN-TR", "Tripura", "16", ()),
    StateRecord("IN-UP", "Uttar Pradesh", "09", ()),
    StateRecord("IN-UT", "Uttarakhand", "05", ("uttaranchal",)),
    StateRecord("IN-WB", "West Bengal", "19", ()),
)


def _normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())


_LOOKUP_BY_NAME: dict[str, StateRecord] = {}
for _rec in STATES:
    _LOOKUP_BY_NAME[_normalize(_rec.region_name)] = _rec
    for _alias in _rec.aliases:
        _LOOKUP_BY_NAME[_normalize(_alias)] = _rec

_LOOKUP_BY_REGION_ID: dict[str, StateRecord] = {r.region_id: r for r in STATES}
_LOOKUP_BY_ST_CODE: dict[str, StateRecord] = {r.st_code: r for r in STATES}


def resolve_by_name(raw_name: str) -> StateRecord | None:
    return _LOOKUP_BY_NAME.get(_normalize(raw_name))


def resolve_by_region_id(region_id: str) -> StateRecord | None:
    return _LOOKUP_BY_REGION_ID.get(region_id)


def resolve_by_st_code(st_code: str) -> StateRecord | None:
    return _LOOKUP_BY_ST_CODE.get(st_code)
