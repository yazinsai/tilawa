# Word-internal nucleus rhythm

Status: needs-human — Modal inspection, tests and training all failed before returning command output. The prototype is committed, but its runtime model remains `null` until a training-only fit succeeds. This is an infrastructure block, not a premise kill.

## Prediction

Tracker −1, overall 0 on practice. Correct localization could remove the false `54:36:1` passage skip in `lab-v1.5-p46/phone` while exposing the documented nearby listener warning at `54:37:2`. No standalone fold gain is claimed.

## Mechanism

The saved audits put the near continuation only 0.3915 edits behind the rhyme restart over the same observed interval. The restart can preserve syllable roles by deleting a vowel nucleus. This experiment tests whether the observed timing fits its resulting word grouping, rather than repairing an accounting defect or changing that deletion's price.

`word-rhythm.ts` compiles each reference word into contiguous vowel nuclei and intervening consonant groups. Pronunciation-expanded vowels belong to one lengthened nucleus; all their existing edit charges remain. A Gaussian Markov copula connects the normalized emitted-run lengths of nuclei assigned to the same word. Parameters depend on short/long nucleus classes and single/multiple intervening consonants, with no lexical or location identifiers. The copula preserves individual duration marginals. Its matched independent control uses precisely the same marginals and zero correlations, discarding word grouping.

Both likelihood partitions traverse the same recorded local graph over the existing 16-token horizon, including insertions, reference deletions and movement arcs from each pre-update winner. Latent nucleus placements and word divisions retain separate probability mass. When a nucleus has no emission, integrating it out multiplies its adjacent Gaussian correlations; this adds no token or free deletion. Insertions retain the last nucleus, whereas a word boundary or movement starts a new group. All candidate endpoints receive the joint-versus-independent contribution symmetrically, using the tracker's existing three-nats-per-edit conversion, when the original graph proposes a local run restart. Published events, checker rules, commitment margins and original acoustic costs are retained.

`word-rhythm-fit.py` selects only explicit `split=train`, `kind=none` help-train records without events, repeats, exclusions or unsupported labels. An eval/test/heldout label excludes its entire speaker. A missing eligible decode stops the fit. Transcript-conditioned edit forward/backward supplies duration and nucleus-pair expectations without hard tracker-visit labels. Three deterministic speaker folds select shrinkage from 10, 100 or 1000 using full marginalized timing likelihood, against identical-data independent-duration controls. Final parameters and an input SHA-256 are written only after fitting succeeds. The design was saved before examining target timing.

## Validation

- Six behavior tests passed locally through Node's TypeScript stripping with a small `node:test` assertion adapter (no dependency installation). These are smoke checks, not the required core/SDK build and Vitest suite.
- Tests check copula marginal normalization by numerical integration, nucleus/group compilation, missing-nucleus correlation products, resets at word boundaries, identical-data grouping versus independent controls, common tempo invariance, finite evidence after edits, missing-timing fallback, search rewind and rho-zero equivalence.
- Python estimator and estimator contract checks compile. Contract checks cover speaker exclusion, cross-language structure, identity-graph mass preservation and independent timing partition mass; execution requires Modal's NumPy.
- `git diff --check` passes.
- Modal request 1 (data/storage inspection) returned `NotFoundError: No such file or directory`, with zero reported compute spend and no command output. This supplies no training or target evidence.
- Modal request 2 (six-shard core/SDK build and tests) and request 3 (training fit) also returned `NotFoundError: No such file or directory`, each with zero reported compute spend and no command output. Neither the required suite nor training ran. No remote data inspection succeeded.
- Practice has not been scored (0 of 3 allowed calls). Target preference reversal and warning wins have not been measured.

## Limits and next validation

The null model means current default engine decisions are unchanged; this is not a fitted experiment result. A successful fit must replace `word-rhythm-model.ts` with its generated frozen parameters before a product replay or supervisor score. Do not select parameters using the target. Run the estimator contract checks alongside the fit, then the core/SDK suite and a controlled target replay with the learned correlations and with the same model's correlations set to zero. Measure the continuation-versus-restart ordering on the identical retained interval and count actual target warning wins. The user's premise rule kills a built change winning fewer than three targets; no full practice score is warranted before that check passes.

Emitted-run lengths are recognition timing observations, not measured true articulation lengths: CTC blanks hide sound boundaries. The model uses joint relative nucleus durations, not acoustic word-boundary posteriors or inter-emission gap probabilities. It assumes Gaussian residual marginals and a Markov dependence within each word; it does not recover missing speech, revise accepted history or certify ayah coverage. Runtime overhead with a fitted model and ordinary-reading control stability remain unmeasured.
