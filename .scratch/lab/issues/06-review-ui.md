# 06 Listening review UI (data tab)

Status: ready-for-agent
Type: task
Blocked by: none

Spec: `../spec.md` decisions 13, 14, 16. Served as a Modal web endpoint; audio is read from the lab volume.

## Acceptance

- A review queue (for v1: 20 splices plus 20 random cases covering clean stretches, repeats and phone-degraded audio). Each item has playable audio around the case, the labelled text, and for splices the before/after source audio.
- Verdicts: OK, or bad data (label / audio / splice), plus a free-text note kept verbatim. Saved to the volume, not browser storage.
- A running tally against the acceptance bar (no more than 1 bad in 40).
- The same item view and flag buttons are reused by the dashboard drill-down (ticket 11).
