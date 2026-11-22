# Audio failure review

Use automated triage before requesting listening work from the user.

1. Select matching student and teacher dumps and record the run, model, context and scoring policy. Group missed labels and unmatched warnings into incidents by recording and location; retain original records.
2. Reuse saved human findings. Inspect error-class, state and word-boundary mismatches, teacher/student evidence, and tracker/checker traces. Separate confirmed findings from hypotheses. Teacher agreement is correlated evidence, not independent verification.
3. Give unresolved audio cases to an independent audio model where an authorized connected service is available. First assess it on human-reviewed calibration clips, including clean speech, omissions and substitutions. Save provider/model, audio hash, prompt, raw response, calibration results and uncertainty. Judge the heard recitation before exposing expected labels to reduce anchoring. A generic transcript cannot establish fine pronunciation correctness.
4. Present the user only residual disagreements, low-confidence cases and a small spot-check sample of provisional agreements. Explain the exact listening question and offer playable audio in small batches. Preserve spoken reviews verbatim alongside their interpretation.
5. Keep automated opinions separate from benchmark truth. Apply label corrections only with review evidence and the project's scoreboard procedure. Report queue counts, resolved reasons and remaining uncertainty.

Existing helper: `tools/help_review_triage.py` groups leadFree dump failures and compares the matching teacher dump. Its output is provisional and does not change labels. Private findings live under `.scratch/help-review/`. A reviewed recording can contain further unresolved incidents; check the scope of each finding.

If no usable independent audio service is connected, finish local trace triage and identify the remaining audio queue explicitly. Do not represent text-only inspection or the existing teacher as an independent listening review.
