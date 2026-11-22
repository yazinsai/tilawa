# Measurement priorities before choosing an engine change

Research note, 30 October 2026. No engine, labels, or benchmark results changed.

## Findings

1. **The session replay and SDK do not use the same acquisition cadence.** `packages/sdk/src/session.ts` creates the tracker with `acquireMs: 1000`; `packages/core/src/bench/acceptance.ts` does too. `packages/core/src/bench/sessions.ts:runSession` omits it and pushes 40 ms cells. `tracker.ts` searches on each push when the option is absent. Thus the expanded session report is evidence of that replay policy, not an exact measurement of the SDK acquisition policy. Compare both policies on identical evidence before tuning lock thresholds. This does not invalidate all tracking results or establish which cadence is better.

2. **Cold-start E0 attribution is a diagnostic, not isolated causal attribution.** The free and declared arms differ in both knowledge of the intended passage and checker policy. A substitution can produce another valid ayah, making the intended passage unknowable from that clip alone. The session spec records that 21 of 25 earlier tracker-attributed misses were such cold-start substitution clips. Do not interpret the current 27/35 bucket as 27 fixable tracker bugs.

3. **Some apparent false warnings are diagnosis/localization errors or incomplete labels.** Seven saved human findings include a correct-location wrong-class warning, two warnings spanning the preceding word and the real substitution, an omitted word absent from the label, two genuine false accusations, and a real passage skip whose boundaries remain unresolved. These are selected cases, not a prevalence estimate. Keep strict event scoring, add an incident-level diagnostic, and correct labels only through the existing review process. Do not credit a more lenient metric as an engine improvement.

4. **Declared passage regressions are concentrated and can be paired.** In `session-qud-int8-expanded-v1.json`, the six assignment recordings produce 99 clean warnings in 110.754 clean minutes (4.469/5 min). Those same recordings at `none` produce 41 (1.851/5 min). The additional 58 warnings are concentrated in Luhaidan/Fatir (+52) and Mandour/Taha (+6). Comparing assignment-six with free-eleven would mix policy and sample composition. This is a useful targeted test of anchoring/occurrence state, not proof that declaring the passage is generally worse.

5. **Existing long-session and clip benchmarks cover different failure modes.** The 743-minute stitched student report gives 1.46 false wrong/added words per five minutes, but excludes omissions/skips, stitches independently decoded clips, and changes voices at many seams. The expanded supported-style QUD report has 11 recordings, 9 speakers and 195.689 clean minutes; it reports 69 clean warnings (1.763/5 min), 79 total warnings, 93.246% located time and 6/11 starts within five seconds. It has no labelled error positives, excludes four hadr recordings, and contains one evidence-mismatch failure. Neither establishes error recall on real learner sessions. Retain both reports and state their scopes.

6. **A representative continuous learner test set is missing.** The existing coverage audit has 388 held-out clips/29 speakers but no minute-long same-speaker consecutive run, no error clip with both same-speaker clean neighbours, and no word-timed help clips. Synthetic mixed-speaker/evidence seams should not substitute for this missing test.

7. **Further tuning needs a genuinely unused selection split.** `modal/student_train.py` repeatedly evaluates `clean-match/test` during training, though checkpoint saving is fixed-schedule, not automated best-test selection. Repeated human comparisons can consume a test set. Tune new policies/objectives on a development split; reserve new speakers/sessions for the final comparison. Existing help eval should remain a regression set, not be relabelled an untouched test after more tuning.

## Recommended experiment contract

- Freeze model weights, evidence, reference version, split, context, streaming cadence, and warning policy for every baseline/candidate pair. Change one intended variable per comparison, or name an explicit factorial experiment.
- Save per-incident changes, not just percentages: warnings removed, real mistakes lost, wrong words localized correctly, new wrong locks, and time spent ambiguous/lost.
- Report clean-warning rate, strict event precision/recall, incident detection, exact location/class, acquisition delay, located time, wrong-position episodes, and confirmed-warning delay separately.
- Never let abstention improve recall: all labelled mistakes stay in the denominator. Also report abstention duration and coverage so silence cannot masquerade as a good tracker.
- Use speaker/session-level paired uncertainty. The current 64 positive errors and 11 long recordings cannot support fine percentage distinctions; clip-level uncertainty also understates within-speaker correlation.
- Require a meaningful clean-warning reduction with no material recall/latency regression before promotion. Keep the existing product goals (at most one false warning per five minutes; precision at least 90%, recall at least 80%) distinct from experimental acceptance criteria and from evidence that those goals have been met.
- For real-session promotion, reuse the existing spec's coverage and two-run agreement requirements rather than inventing looser gates. Start with a smaller reviewed development panel to test mechanisms; it is not a release-validation substitute.

## Sources

- `packages/sdk/src/session.ts` (`start`, `ACQUIRE_MS`)
- `packages/core/src/bench/sessions.ts` (`runSession`, `scoreSessionEvents`)
- `packages/core/src/bench/acceptance.ts`
- `.scratch/session-bench/spec.md`
- `.scratch/session-bench/coverage.json`
- `.scratch/help-review/human-findings.json`
- `.scratch/help-review/automated-triage.json`
- `docs/benchmarks/session-qud-int8-expanded-v1.json` (local untracked report)
- `docs/benchmarks/stitched-warnings-v1.md`
- `docs/benchmarks/acceptance-v3.md`
- `docs/benchmarks/local-window-v1.md`
- `modal/student_train.py`
