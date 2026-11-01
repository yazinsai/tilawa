# 09 Bets on GPUs

Status: ready-for-agent
Type: task
Blocked by: 04, 05

Listener bets mean retraining or new data: Modal GPU jobs through `modal/student_train.py` and the decode path (`modal/lab_decode.py`), producing a new listener's evidence that the scorer reads by `--listener`.

## Scope

- **A bet card adds:**
  - the training recipe;
  - the GPU budget in Modal dollars;
  - a checkpoint plan;
  - a cheap early check (for example phoneme error rate on a fixed practice subset after the first epoch) with a kill threshold.
- **The supervisor** spawns training, polls it, runs the early check, kills a bet that misses its threshold, then decodes the practice set and scores the card with the new listener. The exam is decoded only at fold.
- A budget over the configured limit becomes `needs-human` before any spend.
- An executor that needs a GPU sandbox (Modal Sandbox) gets one through the adapter interface of ticket 05.

## Acceptance

- One real bet runs end to end on a small recipe, killed by its early check or scored, with spend recorded.
