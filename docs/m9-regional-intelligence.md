# M9 — Regional Intelligence and Operational GEFS Contract

## Scope
M9 adds the region-level reliability layer and hardens operational GEFS member handling.

## GEFS member contract
Operational GEFSv12 is expected to contain 31 members:
- `gec00`
- `gep01` through `gep30`

Historical GEFSv12 reforecast remains a separate training contract (5 routine members, weekly 11-member extension). The live system must not pretend that historical 5-member data are equivalent to operational 31-member data.

## Reliability score
`reliability_score` is a 0–100 trust score. Higher is safer. It combines:
1. calibrated bust probability,
2. predicted error relative to its variable threshold,
3. data coverage,
4. ensemble member completeness.

Coverage/member penalties can only lower trust; they cannot artificially increase safety.

## Safety rules
- Invalid operational member IDs are reported.
- A partial ensemble is allowed for diagnostics but must expose member count and cannot be represented as a complete 31-member cycle.
- No observed error is used to produce live confidence.
- Reliability is not a probability and must not be presented as one.

## M9 implementation notes
The live default now requests all 31 operational members. This changes the prior 5-member live default; historical model training remains based on the historical archive contract. Every API/UI consumer should display the observed member count so a partial cycle is never mistaken for a complete ensemble.
