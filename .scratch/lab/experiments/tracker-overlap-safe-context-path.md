# Tracker overlap-safe ledger context

Status: killed on the premise. The existing context ranges cannot duplicate a reference word occurrence. No engine change retained.

## Prediction

Practice lab-v1.5u, phone: tracker −1 and overall −1, with substantial zero-gain risk. Target: `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

## Mechanism

The card proposes replacing independently concatenated tracker anchor context with ordered reference-occurrence intervals, composing overlaps once in both read and skip hypotheses while retaining explicit repeats and identical text at separate locations. Acoustic windows, positions, movement events, edit prices and confirmation margins must stay unchanged.

## Premise audit

The inspected implementation has no tracker-exported ledger context strings. `checker.ts::closeGap` constructs the ledger context directly from reference word indices. Each candidate ayah `a` satisfies `left < a < right`.

- Left context starts at `max(ayahFirst[left], leftSpan.wordFrom)` and ends at `min(ayahLast[left], leftTo)`.
- Candidate context is exactly `ayahFirst[a]` through `ayahLast[a]`.
- Right context starts at `ayahFirst[right]` and ends at `min(ayahLast[right], wordTo)`.

These intervals belong to three distinct ayahs and cannot share a reference word occurrence. A tracker candidate's broader `observed.span` is not independently concatenated into the hypothesis. `ayah-ledger.ts::hypothesis` emits each supplied word once, then adds optional word and ayah repeat arcs. Those arcs do not introduce mandatory duplicate word obligations.

This invariant covers the target candidate at 54:36 as well: its left context is confined to an earlier ayah, its candidate context to 54:36, and its right context to a later ayah. Removing overlapping reference occurrences would therefore leave both hypotheses' labels and repeat arcs unchanged. The proposed defect is absent; no alternative ranking, window, issue-edit or warning-veto mechanism was substituted.

## Validation

- Read `AGENTS.md` and the unmodified experiment card; inspected the sole ledger call site, the reference word range construction, the hypothesis compiler and existing ledger unit tests.
- Added a temporary observation hook at the live ledger call and prepared a target replay to check occurrence multiplicity and hypothesis equality. Requested Modal run 1 through the supplied supervisor command. It failed before execution with `NotFoundError: No such file or directory`, returned no command log, and reported zero compute spend. No runtime observation or likelihood measurement was obtained.
- Requested `python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 --timeout 20 -- 'bash tools/autolab_tests.sh'` as Modal run 2. It failed before execution with the same infrastructure error and returned no command log. Core/SDK builds and tests could not be validated here.
- Removed the temporary engine hook and replay script. `git diff --exit-code` confirmed that `tracker.ts`, `checker.ts`, `ayah-ledger.ts` and `lab-experiment.json` match the starting branch.
- No behavior change remains, so no unit test change was retained. Practice scoring was not requested: 0 of 3 allowed runs used. Targets won: 0; no practice component or overall delta was measured.

## Limits

The kill is supported by the source invariant, not a successful target replay. Modal infrastructure prevented runtime and test confirmation. This audit tests reference occurrence multiplicity, not whether the selected context or its acoustic alignment is correct, and does not explain the previously reported 617-nat residual. Identical pronunciation strings at different Quran positions are separate occurrences; deleting them by string equality would change legitimate obligations.
