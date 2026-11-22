# 08 Portfolio, plateau, novelty and coverage

Status: resolved
Type: task
Blocked by: 04, 06

Spec decisions 7, 8 and 9 as supervisor logic.

## Scope

- **Slot allocation:** components get slots in proportion to practice attribution, at least 1 each. Tiers split them 50/30/20. Rounding goes to the larger remainder. A bet holds its slot for its whole run.
- **Incident clusters** come from the main line's practice cases, keyed by component, error class and mechanism tag where one exists. The coverage map is stored in `state.json`: untouched, tried (with experiment IDs) or fixed.
- **Rules:**
  - at most one `tuning` experiment open at a time;
  - a cluster tried twice without a fold needs a written reason in the card;
  - the plateau trigger and its freeze;
  - the follow-up question to the user after 24 more hours.

## Acceptance

- Unit tests for the allocation with today's attribution (listener 47, checker 37, tracker 5) and 6 slots, the plateau freeze and its reset on a fold or a new test set, and the tuning cap.

## Comments

Built (1 Nov): `tools/autolab_portfolio.py`, tested in `tools/test_autolab_portfolio.py`.

- With today's attribution and 6 slots, the listener gets 3, the checker 2 and the tracker 1. Tiers come out as 3 exploit, 2 explore and 1 bet, and the bet goes to the listener (tested).
- A frozen component's exploit slots become bets.
- Plateau: 5 decided non-folds in a component, or 24 hours with no fold, freezes it. A fold thaws that component and clears the 24-hour freezes. After 24 more hours frozen, the user gets one question. A new test set resets the counters.
- At most one tuning diff can be open at a time.
- Incident clusters are keyed `component/errorClass/missed|false-warning` from the main line's practice phone cases. Coverage is untouched, tried or fixed, and `failed_tries` counts decided non-folds per cluster.
