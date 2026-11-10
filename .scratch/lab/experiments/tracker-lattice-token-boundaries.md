# Local acoustic token-boundary lattice

Status: needs-human — implementation complete; Modal validation blocked before execution.

## Prediction

Tracker −1, overall 0 on lab-v1.5u practice. Recovering the local continuation could remove the tracker false skipped-passage warning at `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`, while exposing the documented nearby listener false wrong-word warning. This is the card's forecast, not a measured result.

## Mechanism

The existing local matcher compares every reference frontier on one hard phone string. `trackerInput` receives posterior rows but previously removed them. It now routes those rows in a separate local-only field, keeping the existing acquisition path decoded-only and preserving the listener's phone objects and the checker's input.

When an in-lock step selects a different run origin, the tracker builds a bounded CTC prefix lattice over the most recent local comparison interval: at most the existing 16 observed phones, with one saved DP checkpoint before that interval. Blank and nonblank prefix endings are separate. A phone continuing without a blank extends its run; the same phone after a blank can create a second articulation. The lattice retains distinct collapsed strings and supported run boundaries. New prefixes require a label listed in the posterior top-k.

Discovery retains 64 strings per frame. Each final string is rescored by the existing exact CTC chain forward sum, which restores all equivalent frame/boundary paths for that string rather than multiplying probability by its number of boundary witnesses. A second witness of the same string is ignored at composition. The 512-frame work bound falls back to decoded tracking for longer intervals; it does not shorten an interval and then compare it against a longer one. Posterior retention is bounded to that older frame history plus the current input chunk.

Each distinct string is replayed from the same saved checkpoint through the existing local reference-edit recurrence. For each reference frontier there is one minimum edit/movement explanation per string, weighted by that string's summed CTC probability. These terms are summed across strings using the existing tracker conversion of 3 nats per plain edit. Alternative edit alignments do not supply additional acoustic probability. Absolute accumulated costs are retained alongside normalized DP rows so every string pays for the whole comparison interval. The resulting frontier costs replace the local row inside its existing reference window, retaining the most likely contributor's origin, edit count and recent-cost evidence. Movement arcs, their prices, tie rules, commitment thresholds, event delivery, and checker decisions are unchanged.

## Validation

Ten lightweight algorithm checks passed using Node's TypeScript transformation and a small assertion adapter for the new unit tests (`lattice-audit/smoke.mjs`). They check exhaustive three-frame CTC accounting, blank-separated repeated sounds, exact retained-string probability after pruning, listed-label support, unchanged single-string edit/movement costs, marginalization, immutable read-only comparisons, checkpoint restoration, application of the composed row, local-only posterior routing, and protection against duplicate boundary witnesses. These checks are not a TypeScript build or the real Vitest suites.

All four supervisor requests failed before their commands executed, returning `NotFoundError: No such file or directory` and reporting zero compute cost:

- Run 1: the supplied target trace command.
- Run 2: core build, initial lattice unit tests, and the target audit.
- Run 3: `bash tools/autolab_tests.sh` with six shards (required core and SDK validation).
- Run 4: core build, both new unit-test files, and the target audit/product replay.

No target evidence, trace, or replay output was obtained. No practice score was requested (0 of 3 used). There is no measured target gain, overall change, studio change, or clean-session result. The card and all scorer/tooling/data files are unchanged.

The diagnostic in `lattice-audit/audit.ts` is ready for a restored Modal environment. It generates an observed tracker copy in `LAB_OUT`, runs decoded-only baseline tracking, compares the target's competing frontiers on 64- and 256-prefix discovery beams, and separately replays the product tracker/checker. It reads existing lab helpers and evidence without changing them. Run it from the repository root with:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run -- 'pnpm --filter @tilawa/core... build && pnpm --filter @tilawa/core exec vitest run test/ctc-prefix.test.ts test/local-acoustic.test.ts && pnpm --filter @tilawa/core exec tsx ../../.scratch/lab/experiments/lattice-audit/audit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- 'bash tools/autolab_tests.sh'
```

## Limits and decision

The mechanism's premise and kill criterion remain untested. Infrastructure failure is not evidence that the target lacks an alternative segmentation, and no zero-win measurement is claimed. The implementation remains on the experiment branch for validation after the supervisor environment is restored.

The lattice conditions on the saved greedy state before its interval, including a blank boundary at its acoustic start. It cannot recover uncertainty before that checkpoint. String discovery is approximate and may omit useful strings; probability accounting is exact for the strings retained. Run boundaries retain a supported representative; all boundaries for the same string share their forward-summed probability because the unchanged reference-edit costs do not read their timing. Product effects, runtime on real sessions, and the known route to an exposed listener warning still require replay.

After restoration, run the full tests and target audit. Kill and revert the engine changes if the stated premise fails or the built replay wins fewer than three targets, as instructed. If it wins at least three, obtain the authorized practice score before deciding; losses elsewhere are the fold gate's decision.
