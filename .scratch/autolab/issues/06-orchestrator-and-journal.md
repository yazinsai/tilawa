# 06 Orchestrator prompts and the journal

Status: claimed
Type: task
Blocked by: 02

The Opus 5.5 calls of spec decision 1. Each is a stateless call: a fixed prompt, a context pack built by code, and a JSON answer checked against a schema. A malformed answer gets one retry, then becomes `needs-human`.

## Calls

- **plan(slots):** given the attribution, the incident clusters with coverage, every past card with its verdict and journal note, and the tier and component of each free slot, write that many cards. Each card has an ID, component, tier, hypothesis, mechanism, target incident IDs, expected counts and a kill criterion.
- **judge(card):** the novelty check of spec decision 8. Accept, or reject with the past card or cards it restates.
- **review(experiment):** given the card, the diff stat, the gate result and the incident deltas, write the journal entry and propose a follow-up card, or say why the line is dead.
- **strategy():** daily and on a plateau. Write the "why stuck" memo, propose a portfolio change within the spec's bounds, and propose bets.

## Journal

`.scratch/autolab/journal.md`, append-only and committed on `autolab`. It holds one entry per reviewed experiment and per strategy call, in the plain style of the existing experiment write-ups. It is the orchestrator's memory, so the context pack includes the most recent entries plus every entry whose target incidents overlap the current ones.

## Acceptance

- Each call has a recorded example input and output under `.scratch/autolab/examples/`, built from the real ledger. The plan call is checked by hand once: on the current ledger it must give the listener at least half its cards.

## Comments

Built (1 Nov): `tools/autolab_brain.py` and `tools/test_autolab_brain.py` (11 tests with a fake model).

- There are five calls: `plan` and `review` on Opus, `strategy` on Opus, and `judge` and `review_diff` on GPT (spec decision 18). Each answer is a JSON block checked against a schema, with one retry that includes the error.
- Card checks:
  - the ID is new across both ledgers;
  - the card takes its slot's component and tier;
  - the hypothesis is one line;
  - the expected values are whole counts;
  - the cluster exists and the targets are incident keys in it;
  - a cluster with 2 or more failed tries needs a `reason`;
  - a bet needs `gpuUsd`.
- The context pack holds the shadow line's practice score and attribution, the clusters with their incident keys and coverage, the history from the human lab ledger and autolab's (so the 24 experiments from 30–31 October count for novelty), and the end of the journal. It has no exam data (tested).
- The prompts live in the module, a protected path.
- Pending: recorded example inputs and outputs from the real ledger under `.scratch/autolab/examples/`, and the hand check that the plan gives the listener at least half its cards. Both need a live key.
