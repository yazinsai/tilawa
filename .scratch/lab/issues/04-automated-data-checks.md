# 04 Automated test set checks

Status: ready-for-agent
Type: task
Blocked by: 03, 05

Spec: `../spec.md` decision 16.

## Acceptance

- Each planted mistake: the teacher's alignment of the built audio hears the planted word and not the original at that spot.
- Each clean stretch: the teacher matches the text (PER under the gold-set teacher level, with a margin).
- Reciter not in any training source (from 01).
- Session ayah timings consistent with the alignment.
- Failures make the builder rebuild or drop the case and are logged with the reason. No case is hand-patched.
- The same checks run on any test set version, including the old help clips (ticket 13).

## Comments

- Checks run with `modal run modal/lab_check.py` (`--test-set`, `--variants`, `--only`, `--no-write`) on teacher evidence, one container per session × variant. Logic is in `tools/lab_checks.py`, tests in `tools/test_lab_checks.py`. Output goes to `/lab/data/lab-v1/checks-v1.json`, with per-case pass/fail and reasons. Method and numbers are in `docs/benchmarks/lab-v1.md#automated-checks`.
- Studio (acceptance):
  - mistakes 357/370 (wrong 112/114, mutashabihat 14/15, omitted 74/78, added 104/110, skipped 53/53);
  - repeats and restarts 40/40;
  - clean stretches 435/450 (pooled PER 2.15% against a bar of 6.78% = 2 × teacher gold 3.39%);
  - timings 32/40 sessions, 10 occurrences off by more than 0.3 s, none a label error on inspection;
  - reciters 40/40.
- Phone: mistakes 345/370, clean 398/450 (PER 3.99%), timings 23/40. Failures there are mostly the teacher on degraded audio.
- Nothing is hand-patched. The failures are low (3.5% of mistakes) and marked rather than dropped:
  - The scorer reads `checks-v1.json` beside the manifest (`--checks`). It tags matching cases with `dataCheck` and leaves counts unchanged; tested in `lab.test.ts`.
  - `review/v1-check.json` now holds every failed splice: 13 swapped in for the last passing splice of the same kind and class, ids otherwise stable. `v1-check.base.json` keeps the builder's queue.
  - Rerunning the build's plan stage rewrites the queue, so rerun the checks after it.
- Systematic, for the builder (ticket 03). Not applied, so v1 stays as built for ticket 07:
  1. Donors followed by a pause in their source are labelled with the continuous-form tail. This affects 16 of 239 donors and causes 3 of the 13 edit failures. When the donor's source boundary is a pause, `said` should use the pause (`iso`) tail.
  2. Natural repeats inside host ayahs are unlabelled, because the source alignment's garbage blank absorbed them. This causes 12 of the 15 clean failures and e11-x9. Either refuse ayahs whose teacher decode is about 5% longer than the text, or label the repeat.
  3. Weak fricative or pharyngeal donor heads at linked seams: حين at e18-x7 and e18-x10 loses its ح. Refuse them or require a pause seam.
- Open against acceptance:
  - "Failures make the builder rebuild or drop": the builder doesn't consume `checks-v1.json` yet. With fixes 1–3 that becomes a rebuild to lab-v1.1, so the call is ticket 07's.
  - The help clips (ticket 13) aren't in the lab session format yet. The checks take any lab test set (`--test-set`), but the help clips need an adapter.
  - Status stays open for those two items.
- No scoreboard row: nothing in the scoreboard reads lab-v1.

## Comments

- Checks run with `modal run modal/lab_check.py` (`--test-set`, `--variants`, `--only`, `--no-write`) on teacher evidence, one container per session × variant. Logic is in `tools/lab_checks.py`, tests in `tools/test_lab_checks.py`. Output goes to `/lab/data/lab-v1/checks-v1.json`, with per-case pass/fail and reasons. Method and numbers are in `docs/benchmarks/lab-v1.md#automated-checks`.
- Studio (acceptance):
  - mistakes 357/370 (wrong 112/114, mutashabihat 14/15, omitted 74/78, added 104/110, skipped 53/53);
  - repeats and restarts 40/40;
  - clean stretches 435/450 (pooled PER 2.15% against a bar of 6.78% = 2 × teacher gold 3.39%);
  - timings 32/40 sessions, 10 occurrences off by more than 0.3 s, none a label error on inspection;
  - reciters 40/40.
- Phone: mistakes 345/370, clean 398/450 (PER 3.99%), timings 23/40. Failures there are mostly the teacher on degraded audio.
- Nothing is hand-patched. The failures are low (3.5% of mistakes) and marked rather than dropped:
  - The scorer reads `checks-v1.json` beside the manifest (`--checks`). It tags matching cases with `dataCheck` and leaves counts unchanged; tested in `lab.test.ts`.
  - `review/v1-check.json` now holds every failed splice: 13 swapped in for the last passing splice of the same kind and class, ids otherwise stable. `v1-check.base.json` keeps the builder's queue.
  - Rerunning the build's plan stage rewrites the queue, so rerun the checks after it.
- Systematic, for the builder (ticket 03). Not applied, so v1 stays as built for ticket 07:
  1. Donors followed by a pause in their source are labelled with the continuous-form tail. This affects 16 of 239 donors and causes 3 of the 13 edit failures. When the donor's source boundary is a pause, `said` should use the pause (`iso`) tail.
  2. Natural repeats inside host ayahs are unlabelled, because the source alignment's garbage blank absorbed them. This causes 12 of the 15 clean failures and e11-x9. Either refuse ayahs whose teacher decode is about 5% longer than the text, or label the repeat.
  3. Weak fricative or pharyngeal donor heads at linked seams: حين at e18-x7 and e18-x10 loses its ح. Refuse them or require a pause seam.
- Open against acceptance:
  - "Failures make the builder rebuild or drop": the builder doesn't consume `checks-v1.json` yet. With fixes 1–3 that becomes a rebuild to lab-v1.1, so the call is ticket 07's.
  - The help clips (ticket 13) aren't in the lab session format yet. The checks take any lab test set (`--test-set`), but the help clips need an adapter.
  - Status stays open for those two items.
- No scoreboard row: nothing in the scoreboard reads lab-v1.
- The provisional baseline (`docs/benchmarks/lab-baseline-v1.md`) reads the checks. 10 practice cases carry `dataCheck`; counts are unchanged.
