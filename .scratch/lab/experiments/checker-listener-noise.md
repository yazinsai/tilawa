# Checker tolerance for listener noise

Status: open; awaiting the user's fold or discard decision.

- Experiment: `checker-listener-noise`
- Test set: `lab-v1.4.v3`, practice only (20 sessions, 88.136 recited minutes, 134 planted mistakes)
- Main line: `975702f`; baseline run `main-975702f-20261030T140816Z`
- Scored implementation: `bcd9cfc`
- Result: `exp-checker-listener-noise-bcd9cfc-20261030T153250Z`
- One scored iteration. Replays, build, tests and scoring ran on Modal. No exam scoring or case inspection; no fold or discard.

## Prediction and result

The card predicted five fewer overall wrong calls and zero change in checker-attributed wrong calls: most target failures were charged to the listener by the perfect-transcript counterfactual. The prediction was recorded before scoring.

| Practice phone metric | Main | Experiment | Change | Prediction |
| --- | ---: | ---: | ---: | ---: |
| Overall wrong calls | 19 | 14 | -5 | -5 |
| Missed mistakes | 16 | 11 | -5 | |
| False warnings | 3 | 3 | 0 | preserve |
| Listener-attributed wrong calls | 9 | 3 | -6 | most gains here |
| Tracker-attributed wrong calls | 2 | 3 | +1 | |
| Checker-attributed wrong calls | 8 | 8 | 0 | 0 |
| Checker with oracle evidence and true position | 14 | 10 | -4 | preserve or improve |

Overall wrong calls per five minutes fell from 1.078 to 0.794. The oracle checker improved from 12 misses + 2 false warnings to 10 misses + 0 false warnings (0.794 to 0.567 per five minutes). Oracle misclassifications fell from 6 to 4; phone engine misclassifications remained 6. Attribution and oracle component performance are different measures: the unchanged checker attribution does not mean the oracle checker was unchanged.

The listener output was unchanged (1,681 phoneme errors, PER 0.0425). The tracker still had eight false jumps. The extra tracker-attributed warning is a changed checker reaction to an existing incorrect lock.

Studio practice also improved, from 20 wrong calls (16 misses + 4 false warnings) to 14 (10 + 4). Its new regressions are listed below.

## Diagnosis: true-position listener evidence versus oracle evidence

Instrumented the original checker with its alignment operations, word visits, candidates, support margins and confirmations. Compared the `truePosition` arm (correct position, app listener) with oracle evidence; phoneme inspection used `tools/lab_phonemes_core.py`. This isolates checker decisions from tracker placement.

### 1. Incomplete word visits and vowel residue hide or misplace omissions

- `lab-v1.4-p09/phone/mistake:omitted_word/19:82:1/315.75` — **fixed**.
- `lab-v1.4-p09/phone/warning:omitted_word/19:81:8/317.08` — **removed**.

The listener shortened the doubled z in عِزًّا (`عِززَاا` → `عِزَاا`) and changed the next word's initial s to f. It heard no كَلَّا. The old alignment left عِزًّا after only 3 of 7 reference phonemes, below `MIN_COVERAGE = 0.5`, and inserted its vowel tail after the jump. This joined the previous word and كَلَّا into an omission beginning at the wrong word. Oracle evidence visited the whole previous word and correctly blamed كَلَّا. Charging a shortened double as a held sound keeps the previous word intact.

- `lab-v1.4-p12/phone/mistake:omitted_word/77:41:3/165.81` — **fixed**.

The missing فِى borrowed the first f from the noisy following word (`ظ...` heard as `ف...`). The old visit consumed one observed consonant but covered the four-phoneme reference through deletions (cost 7, zero hard errors, one kept consonant), so it counted as spoken. Oracle evidence correctly skipped it. Requiring at least two heard phonemes for a multi-phoneme consonant-bearing word rejects this isolated visit.

- `lab-v1.4-p16/phone/mistake:omitted_word/54:28:6/98.38` — **still missed**.

The missing كُلّ borrowed `فِل` from a much noisier following شرب (`فِلمِمم`). This consumed 3 of 5 reference phonemes, cost 13, one hard error and one kept consonant. The following word's remaining fragment became an uncertain wrong-word candidate (confidence 0.58484 < 0.6). Oracle evidence skipped كُلّ correctly. A two-phoneme minimum cannot resolve this longer borrowed fragment; it needs better joint word segmentation.

- `lab-v1.4-p19/phone/mistake:omitted_word/51:25:6/44.17` — **fixed**.

The preceding سلاما ended in the listener's vowel tail `َاا` instead of the canonical linked ending. That tail landed in the omitted قال interval. Although it retained no consonants, its three phonemes met the old “spoken differently” gate (`max(MIN_ADDED = 3, expectedLength / 2)`). It was not wrong enough to warn, so the omission disappeared. Oracle evidence correctly reported the gap. Requiring consonant evidence before that branch preserves the omission.

### 2. Lengthened added words take alignment detours

- `lab-v1.4-p12/phone/mistake:added_word/77:13:1/38.50` — **fixed**.
- `lab-v1.4-p12/phone/mistake:added_word/77:16:1/51.88` — **fixed**.

The listener heard added فَءِن as `فَءِننن`. Paying the ordinary insertion cost for its repeated ending made detours through other reference words cheaper. The first case produced no candidate; the second proposed at 61.4 s and confirmed at 64.08 s, too late for the labelled insertion at 51.88 s. Oracle evidence proposed and confirmed both on time. Discounting adjacent repeated observed phonemes keeps the added fragment together and confirms both.

- `lab-v1.4-p19/phone/mistake:added_word/51:33:1/102.06` — **still missed**.

The listener heard added وَمِن as `وَمِںںں` and the next لِنُرسِلَ as `دِلُرفِلَ`, with several consonants changed. Alignment borrowed a later reference word for the addition and returned to the corrupted next word. Its wrong-word support margin was 7.37, below the doubled threshold of 20 for uncertain changed sounds. Oracle evidence retained the insertion (margin 13.75 > 10). The repeated-sound change alone does not fix the corrupted next word.

### 3. A correctly located insertion lacks acoustic support

- `lab-v1.4-p10/phone/mistake:added_word/26:164:1/319.30` — **still missed**.

The listener omitted the initial k of added كريم and heard `َرِۦۦم`. The insertion candidate was placed correctly, but its CTC support margin was only 1.63, below the 10-nat confirmation threshold. Oracle كريم had margin 20.22. No global confirmation threshold was lowered for this experiment.

## Implementation

Only `packages/core/src/checker.ts` changed production behavior:

1. Reference deletions inside any doubled sound use the existing held-sound cost (`HELD = 2`), matching how shortened doubles are judged.
2. Adjacent repeated decoded sounds can be inserted at the held-sound cost, including at word boundaries. Unrelated inserted sounds retain their original costs.
3. A multi-phoneme word containing consonants must consume at least two observed phonemes, as well as pass the existing coverage and kept-consonant conditions, to count as visited.
4. Vowel-only gap residue cannot enter the “spoken differently” branch and erase an omission.

No per-case lookup, new acoustic threshold, listener change, tracker change, scorer change or test-set change. Wider insertion discounts were rejected during unit-test development because they disturbed self-corrections or deliberate jumps; they were not scored as another iteration.

## Gains outside the nine calls, and regressions

One extra phone miss was fixed:

- `lab-v1.4-p19/phone/mistake:omitted_word/51:42:1/152.44`.

Two new phone wrong calls offset part of the seven removed calls, giving the net improvement of five:

- **New miss:** `lab-v1.4-p15/phone/mistake:added_word/38:4:6/38.61`. Added إِن was heard as `ءِننن`, and the following وَقَالَ lost its initial w. Previously the insertion candidate `ءِنن` confirmed at 41.60 s. The new alignment absorbs its n sounds into the following word (`نننَقَاالَ`, cost 14, one hard error), so there is no added-word candidate. The oracle already missed this case before the change (support margin 8.67 < 10) and still does; this is an engine regression, not a newly introduced oracle failure.
- **New false warning:** `lab-v1.4-p01/phone/warning:wrong_word/25:57:14/8.68`. The tracker was already locked at the wrong place (`trackerOnPlace = 0`). Both implementations proposed a wrong word there. The new alignment gets enough following onset evidence to confirm at 11.2 s before the lock ends at 11.64 s; the original remained uncertain. Its support margin is 37.23 because the audio does disagree with that incorrect reference. The true-position and oracle arms produce no corresponding error. This is why tracker-attributed wrong calls rise by one despite unchanged tracker output.

Studio has the corresponding new p01 wrong-word warning and a new miss at `lab-v1.4-p19/studio/mistake:added_word/51:33:1/102.06`.

No false-warning count increased in any fully clean practice session (`p02`, `p04`, `p11`, `p17`, `p18`) in either audio variant. The first four remained at zero; p18 retained its existing one. This does not mean there was no new warning on clean speech: the p01 warning above occurs during a clean passage in a session containing other mistakes.

The unresolved cases and p15 regression require more careful segmentation or support modeling. They are left visible instead of adding case-specific exceptions or relaxing global confidence thresholds.

## Validation and reproducibility

- Modal TypeScript build: passed.
- Modal checker unit suite: **51 tests passed**, two files. Six cases in the new `packages/core/test/checker-noise.test.ts` cover shortened doubles near a gap, repeated endings inside additions, isolated borrowed consonants, vowel-only residue, and a clean shortened-double control. Existing tests cover self-correction, deliberate jumps, uncertain evidence and confirmation behavior.
- Command: `pnpm --filter @tilawa/core build && pnpm --filter @tilawa/core exec vitest run test/checker.test.ts test/checker-noise.test.ts`.
- Unit run: `noise-unit-final-631f944-dirty-20261030T153109Z` (same source subsequently committed as `bcd9cfc`).
- `git diff --check`: passed.
- Lab CLI: `python3.12 tools/lab.py run /Users/rock/ai/projects/tilawa-lab-exp-checker-listener-noise`.
- Python 3.12 was used because the Mac's default Python cannot parse an existing lab-ledger f-string. No compatibility edits were made.

Diagnostic artifacts remain on the `tilawa-lab` Modal volume:

- `/runs/noise-diagnose-975702f-dirty-20261030T152326Z/{0,1,2,3,4}/probe.json` and `heard.json`: original true-position/oracle comparison for p09, p10, p12, p16, p19.
- `/runs/noise-regression-probe-975702f-dirty-20261030T153739Z/{0,1}/probe.json`: before/after engine, true-position and oracle comparison for p01 and p15.
- `/runs/exp-checker-listener-noise-bcd9cfc-20261030T153250Z/score.json`: measured practice result.

The scored source stays at `bcd9cfc`; this report is a subsequent documentation-only commit. The ledger records the experiment as open. Main line `lab` remains at `975702f`.
