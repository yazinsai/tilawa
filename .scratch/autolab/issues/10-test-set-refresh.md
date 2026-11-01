# 10 Test set refresh

Status: ready-for-agent
Type: task
Blocked by: 04

Spec decision 10. The ticket 20 builder (`modal/lab_build_sessions.py`), Soniox triage (`review_api.py::triage_solo_main`) and the spot-check queue already exist. This wires them into the supervisor.

## Scope

- **Trigger:** every N folds (default 8) or weekly. The new version keeps the current sessions frozen and adds new ones per split, as lab-v1.5 did. Only held-out reciters are used, each kept in its split.
- **Pause scoring of new experiments** while the build and triage run. Queue the spot check and notify.
- **On your acceptance,** register the version (`lab baseline --manifest ... --exclude ...`), re-score the main line and open experiments, and reset the plateau counters.
- **The practice-to-exam gap** after each refresh goes in the journal: how much of the last period's practice gain held on the new sessions.

## Acceptance

- One refresh runs end to end up to the spot-check queue on a small build (a few sessions per split).
