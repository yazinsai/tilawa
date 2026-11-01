# Autolab: the lab running itself

Status: ready-for-agent

Agreed 1 November 2026. Builds on the lab (`.scratch/lab/spec.md`); terms follow `GLOSSARY.md`.

## Problem

The lab works: about 21 experiments ran in roughly 36 hours on 30–31 October, 8 were folded and 13 discarded, and main-line practice wrong calls on lab-v1.5 fell from 107 to 89. But a person sits in the middle of every step. They pick the next idea, start agents, watch Modal runs, read verdicts, run `lab fold` or `lab discard`, and decide where to look next. That person is the bottleneck and the only thing that stops the lab running overnight.

The history also shows how an unattended loop would go wrong:

- **Local maximum.** All 8 folds were checker or tracker experiments, and every listener attempt failed (`listener-prefix-beam`, `-channel-adapt`, `-channel-blend`, `-phone-channel`, and the blank, paired and targeted attempts in `listener-three-way.md`). Practice attribution is now listener 47, checker 37, tracker 5. A loop that chases cheap wins keeps tweaking the checker and never touches the biggest bucket.
- **Overfitting the practice set.** Agents read the 100 visible practice sessions to design fixes. Hill-climbing on them around the clock will fit those sessions, not recitation.
- **Folds by chance.** A noise threshold of 2 is fine for a person weighing a handful of experiments. Over dozens of attempts a day, some will clear it on luck.
- **Silent stalls.** A `modal run` started from a shell call dies when the call returns (AGENTS.md, compute gotchas). An unattended runner has to survive that.

## Goal

A loop that runs around the clock on the Modal resources the lab already uses. It proposes experiments, implements them, scores them, folds those that clear a strict gate, discards the rest, and keeps exploring. You see progress on the existing Modal dashboard and are asked only for what needs a person:

- listening (bad-data flags, spot checks of a new test set version);
- a revert window on folds (a daily digest, one command to undo a day);
- approval of expensive bets (over a set Modal or LLM spend).

Non-goals: replacing the scorer, the test sets or the ledger; changing how mistakes are planted; the release scoreboard; pronunciation.

Anti-goals, a failure even if the loop technically runs:

- It folds changes whose gain lives only in the practice sessions (leave-one-reader-out or the exam disagree).
- Its experiments collapse into parameter twiddling on one component.
- A model writes or reports a score, or edits the scorer, the test data or the gate.
- It stalls for hours with nothing on the dashboard saying so.

## Decisions

1. **Three roles.**
   - The **supervisor** is deterministic Python (`tools/autolab.py`). It owns the experiment state, git, Modal scoring, the ledger, the fold queue and the gate.
   - The **orchestrator** is Claude Opus 5.5, called only at decision points: plan, review, strategy and novelty judging. Each call is stateless and built from a curated context pack. Its memory lives in files (the ledger and `.scratch/autolab/journal.md`).
   - **Executors** are Grok 4.7 or GPT-5.6 Sol. They implement one experiment card in one worktree. Model IDs come from `Cursor.models.list()`, not hardcoded. The proposer and the judge come from different model families.
2. **Models never score.** The supervisor runs scoring itself, from a clean checkout of the branch tip. An executor's own numbers are notes, not results.
3. **State machine.** An experiment moves through `proposed → accepted → implementing → validating → scoring → gated → folding`, then ends as `folded`, `discarded`, `superseded`, `needs-human` or `abandoned`. Each state has a timeout. The supervisor reconciles: every tick it reads the state, advances what can move and writes it back, so a crash plus restart resumes where it left off. State lives in `/lab/autolab/state.json` on the tilawa-lab volume. Every transition also appends a line to `/lab/autolab/events.jsonl`.
4. **Unattended scoring.** `lab_run` is deployed as a Modal app and called with `modal.Function.from_name(...).spawn()`, polled by function-call ID. The lifetime of a local `modal run` process no longer matters. `tools/lab.py` keeps its CLI and gains `--json` output plus a non-interactive mode (it never prompts).
5. **The fold gate is code** (`tools/lab_gate.py`, pure functions with tests), not an LLM judgment. It was calibrated on the 30–31 October decisions (`.scratch/autolab/backtest.md`, ticket 02). A fold needs every one of these on the practice set, measured after rebase against a freshly scored main line:
   - **Gain:** overall improves by at least 3 wrong calls, and the card's component gets no more than 2 worse. A fix can show up in another component's attribution: `checker-listener-noise` was overall -5, checker +0, as its card predicted. A diff the classifier labels `tuning` (decision 8) needs at least 5.
   - **Spread:** leaving out the reader who gained most still leaves an overall gain of at least 1. Readers are what the exam holds out.
   - **No collateral damage:**
     - no new false warnings on fully clean sessions;
     - the studio slice gets no more than 2 wrong calls worse;
     - real users' clean recitation gains no more than 1 false warning;
     - incidents lost (new misses and new false warnings, matched as in `lab_compare_incidents.py`) are at most 30% of incidents won;
     - a trade, meaning any incident lost, needs a net gain of at least 5. Every fold you made either lost nothing or netted 5 or more; both discards that traded netted 3.
   - **Reported, not blocking:**
     - A session bootstrap. Scoring is deterministic, so 3 incidents won and 0 lost is a real change on these sessions, yet no session bootstrap clears 0 when only 3 of 100 sessions move.
     - Leaving out one surah. Surah 26 is 23 of the 78 practice sessions that have mistakes.
     - Generalization is left to the exam tripwire and the test set refresh (decision 10).
   - **Hygiene:**
     - the diff touches no protected path (the scorer and lab code under `packages/core/src/lab/**`, `tools/lab*.py`, `tools/autolab*.py`, `modal/lab_*.py`, the gate, test data and manifests);
     - the diff adds no literal session ID, reader ID or surah:ayah reference to production code;
     - the TypeScript build and tests pass.
   - **No exam at fold time** (your call, 1 Nov). Autolab folds run `lab fold --practice-only`. The exam is scored only at test set refreshes (decision 10), never per fold, so it can't leak into selection one bit at a time. No exam count appears in the committed summary, the orchestrator's context or an executor's worktree. `lab_gate.exam_tripwire` stays for the refresh check.
6. **Shadow first, then automatic.** Folds land on the branch `autolab` (rebased onto `main` each day), and you fast-forward `main` from it. Once about 10 consecutive gate decisions match what you would have decided, folds go straight to `main`, with a daily tag `autolab/YYYY-MM-DD` for one-command reverts. A backtest of the gate on the 30–31 October experiments (ticket 02) runs before any of this.
7. **A portfolio, not a greedy queue.**
   - Each component gets experiment slots in proportion to its share of practice attribution, with at least one slot each.
   - Within that, three tiers have fixed shares of the slots: **exploit** 50% (a bounded fix to one incident cluster, hours), **explore** 30% (a new mechanism, at most a day), **bet** 20% (retraining, new training data, decoder or architecture changes, days of Modal GPU).
   - A bet's GPU spend over the set limit becomes `needs-human`.
8. **Anti-twiddling.**
   - **Cards target incidents.** Every card names the incident IDs it targets, taken from the main line's practice cases, along with a mechanism and its predicted counts, preregistered before implementation as now.
   - **Diff classifier.** A diff that changes only numeric constants is `tuning`. At most one `tuning` experiment can be open at a time, and it faces the higher gain bar.
   - **Novelty judge.** Before a card is accepted, a model compares it with every past card in the ledger and its journal note. A restated discarded mechanism is rejected.
   - **Coverage map.** Each incident cluster is marked untouched, tried and failed (with links to the mechanisms tried), or fixed. Picking a cluster that has failed twice needs a written reason.
9. **Plateau trigger.** Five consecutive experiments in a component without a fold, or 24 hours with no fold anywhere, freezes that component's exploit tier. The orchestrator then writes a "why stuck" memo to the journal and proposes bets. If 24 more hours pass without a fold, it asks you one question.
10. **Test set refresh.** The real cure for fitting the practice set is new sessions. Every N folds, or weekly, the supervisor builds the next synthetic version with the ticket 20 builder, runs Soniox triage, and queues your spot check. On acceptance it registers the version, re-scores the main line and open experiments, and resets the plateau counters. The refresh is also the one time the exam is scored. The main line's exam result against the previous refresh's goes to you, not to the orchestrator. If the exam got worse by more than 2 while practice improved, folding pauses until you look.
11. **Spend caps.** There is a daily cap on LLM spend and one on Modal spend. At a cap, the supervisor finishes running work and starts nothing new. Spend shows on the dashboard.
12. **Notifications** go through one channel (ntfy, Slack or email; to be picked in ticket 07). They cover the daily digest, `needs-human`, a stale heartbeat, spend caps and paused folding.
13. **Dashboard.** An Autolab tab on the existing Modal dashboard (`modal/lab_review.py`, `modal/lab_web/`) reads `/lab/autolab/state.json` and `events.jsonl`. It shows:
    - agents running now, with state and age;
    - the queue;
    - recent verdicts, with gate details and links to journal entries;
    - each component's practice trend over time;
    - the portfolio split, the coverage map and spend against the caps;
    - the supervisor's heartbeat, red when stale.
14. **Where it runs.**
    - **Phase A:** the supervisor runs on an always-on machine (the Mac under `caffeinate`, or a small box), with executors as Cursor SDK local agents in worktrees. Scoring is on Modal, as now.
    - **Phase B:** the supervisor becomes a Modal cron function and executors become Cursor Cloud Agents on `exp/*` branches in `origin`, so the Mac can be off.
    - Bet-tier executors that need GPUs may run in Modal Sandboxes.
    - **Executor isolation** (your call). Executors get their own Modal environment and token, with a practice-only copy of the test data and evidence. They cannot read the exam or write to `tilawa-lab`; only the supervisor holds that token. Their test runs go through the same `lab_run` code deployed in their environment.
    - The supervisor talks to executors through one adapter interface (start, poll, cancel, collect), so moving between phases means swapping the adapter.
15. **AGENTS.md** changes once the gate replaces the person: "The user decides `lab fold` or `lab discard`" becomes "the autolab gate decides; the user can revert". Until then the shadow branch keeps the current rule true for `main`.

16. **Listener experiments need fresh evidence, and the supervisor makes it.** A listener card changes the decoder, a checkpoint or the training. Its `validating` step has the supervisor run the decode (`modal/lab_decode.py`) from the branch tip on the practice set. The evidence goes to a listener name keyed by the branch sha, and the card is scored with `--listener <that name>`. Executors never write evidence. A listener fold changes the ledger's main-line listener (a new ledger field, not the `DEFAULT_LISTENER` constant), so it needs no edit to a protected path. A fold does not ship the model into the product (your call): that happens at release, against the scoreboard. Exploit and explore listener cards therefore work from ticket 04 on; only training bets wait for ticket 09. *(Builder's call.)*
17. **Shadow mode in detail.**
    - Autolab keeps its own ledger, `/autolab/ledger.json` (ticket 01).
    - You can keep folding to `main` by hand. The supervisor rebases `autolab` onto `main` at the start of each day and re-scores its main line.
    - You take autolab's folds with a fast-forward of `main` to `autolab` right after that rebase, or by cherry-picking single folds.
    - If you disagree with a shadow fold, it is reverted on `autolab`, and every later fold still unmerged is re-gated against the new tip through the normal fold queue.
    - *(Builder's call.)*
18. **Judges are another model family.** Opus writes the cards. The novelty judge (decision 8) runs on GPT-5.6 Sol, and so does an independent review of each executor's diff before scoring. That review looks for special-casing of benchmark cases that the literal check misses. *(Builder's call.)*
19. **Billing.** Opus and the executors all run through the Cursor SDK, so there is one key and one bill. The daily caps are first enforced as a number of agent runs and agent wall-clock hours per day, and Modal spend comes from Modal's usage data. A dollar cap replaces the run cap if the SDK reports cost per run, which ticket 05 will check. *(Builder's call.)*
20. **Executors may score practice, within a budget.** Through the supervisor, never through `lab run` directly, an executor may score the practice set up to 3 times per card. Every attempt is logged on the card, and the gate's write-up shows the attempt count. This replaces "never `lab run`" in ticket 05. *(Builder's call.)*

## Order

1. Ticket 01 (`lab.py` unattended) and ticket 02 (the gate and its backtest) come first and are independent.
2. Then ticket 03 (spawned scoring), ticket 04 (the supervisor and its state), ticket 05 (the executor adapter, phase A) and ticket 06 (orchestrator prompts and the journal). Shadow mode starts once 03–06 are done.
3. Then ticket 07 (dashboard tab and notifications) and ticket 08 (portfolio, plateau, novelty and coverage).
4. Then ticket 09 (the bet tier on GPUs), ticket 10 (test set refresh), ticket 11 (switch to auto-fold on `main`) and ticket 12 (phase B).
