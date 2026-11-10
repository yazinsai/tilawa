# Single-witness word verification

Status: killed on the implementation premise. Engine code and experiment card unchanged.

## Prediction

Card `listener-single-witness-verification`, listener, explore. Expected listener
−1 and overall −1, removing one false warning with no predicted losses on
`lab-v1.5u`, using `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`.

Registered targets:

- `lab-v1.5-p55/phone/warning:wrong_word/26:174:8/107.08`
- `realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`

## Proposed mechanism and source audit

The hypothesis requires independently maximized verification fields: one heard
pronunciation or interval passes lexical eligibility or the surplus-consonant
escape, while another supplies the acoustic margin. The proposed change would
retain and evaluate complete immutable witnesses and transport a passing witness
through ordinary listener evidence.

That verifier is absent from this branch, audited at `d78d06f2`:

- `packages/sdk/src/listener.ts` emits greedy phonemes, their confidence and
  frame bounds, and optional posterior rows. It neither selects word rivals nor
  returns eligibility or word-support fields.
- `packages/sdk/src/gap-decode.ts` replaces accepted gap phonemes and posterior
  rows. It does not retain competing word-verification interpretations.
- `packages/sdk/src/seam-insertion.ts` supplies the existing auxiliary evidence
  for added words. `SeamInsertionEvidence` binds one observed spelling and
  interval to its teacher marginal. There is no wrong-word verification
  certificate in `packages/core/src/evidence.ts`.
- In `packages/core/src/checker.ts`, `Issue` retains `observed`, `obsFrom` and
  `obsTo`, with an explicit `supportRange`, `pair` or `phrase` when needed.
  Ordinary per-word issues are constructed from one visit's token slice and
  tagged `stringPath`. `supportNeed(issue)` tests that issue's heard slice for
  spelling and consonant count; `clarity` and `changedClarity` read its same
  token bounds. `supported(issue)` constructs and scores the stored observed
  sequence over the issue's acoustic comparison window. It returns a boolean
  or null, not independently maximized fields from several pronunciations.
- The lexical candidate paths also retain one heard sequence and its bounds.
  Where a dropped-ending alternative is selected, `best` keeps the chosen form,
  edit cost and matches together; the resulting issue's `observed` and `pair`
  use that form, and acoustic support scores that stored pair. Parallel
  selection keeps the path, span, margin and required bar together.
- `packages/core/src/checker-lexical.ts` enumerates reference pronunciations
  while testing one fixed heard string. `surplusWord` returns a spelling, and
  `alternative` returns eligibility. Neither pools acoustic scores from heard
  alternatives. Held-sound and edge normalization are existing spelling rules,
  not independently selected heard witnesses.

The existing `heardUnion` field reports absorbed tail speech while deliberately
keeping the original issue's confirmation evidence. It is not substituted into
`supportNeed` or `supported` as another source of eligibility or acoustic support.

Thus the hypothesized existential mismatch between independently maximized
heard-pronunciation or interval fields cannot occur in the current verification
implementation. Adding a new verifier or restricting existing spelling rules
would test a different mechanism. No engine change was made.

## Validation and limits

Modal request 1 attempted the supplied trace CLI on both targets, with the card's
listener. Request 2 attempted a diagnostic-only source-instrumented replay of
both targets through the free engine and true-position checker, recording each
issue's heard string, token bounds, lexical result, confidence, acoustic window,
sequence scores and required bar. Instrumentation was confined to ignored
`.autolab` diagnostic files and the disposable remote snapshot; local engine
sources were never edited.

Both supervisor requests failed before command execution with
`NotFoundError: No such file or directory`, returned USD 0, and provided no
shard logs or target measurements. The infrastructure failure is not the kill
reason. The source audit establishes absence of the required field-pooling
mechanism, but does not establish either target's exact runtime margin, warning
route, or an independent listening judgment.

Runtime targets tested: 0. Targets won: 0 (no intervention was applied).
Practice scores used: 0 of 3. No overall or component score change was measured.
No unit tests were added or run because no engine behavior changed. No scorer,
lab tooling, test data, manifests, evidence or card was changed, and no fold was
performed.

Reopen only if an actual word verifier is introduced that independently pools
eligibility and acoustic support across heard pronunciations or intervals, with
a target trace showing an accusation unsupported by any complete witness.
