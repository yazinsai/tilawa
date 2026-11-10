# Direct soft-phone matching relations

Status: killed on the matching-representation premise; experiment card unchanged.

## Prediction

On lab-v1.5u with listener
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`, predict one fewer listener
wrong call and one fewer overall wrong call, with no predicted losses. The sole
registered target is
`realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`.

## Proposed mechanism

Preserve soft phone matches as direct, directional pair relations instead of
equivalence classes. A lexical rival must use its own permitted reference-to-heard
edge; it must not inherit an unsupported match through an intermediate phone.
Retain original phone identities, existing direct-pair costs, and all downstream
confirmation rules. Primary decoding, tracker input, and parallel-graph
confirmations would be unchanged.

The card's kill criterion is that the target's accepted rival uses no match
admitted solely through transitive closure. The task also requires rollback if a
built intervention wins fewer than three targets; there is only one registered
target, so the premise audit precedes implementation.

## Source audit

At base commit `42b61c36c442324eef25cba632f4a55f40827b98`:

- `packages/core/src/fuzzy.ts` already prices `subCost(ref, obs)` by direct
  membership in the declared ordered-pair set. The eight consonant pairs retain
  their existing cost of 5; all other unequal consonants cost 10. There is no
  union-find structure or transitive-closure lookup. The declarations contain
  the chain ث–س–ص, but ث↔ص remains a hard edit. Noon has no soft partner.
- `packages/core/src/checker-lexical.ts` retains original consonants in its
  vocabulary and spelling comparison. `matches` and `alternative` use exact
  strings after held-duration collapse. `surplusWord` first indexes by an exact
  consonant skeleton, and `same` rejects every unequal consonant. Its remaining
  vowel comparisons call the direct `subCost` function.
- `ending`, `droppedSuffixes`, and hard-consonant edit counting also retain phone
  identity. Tail comparisons use reference-to-observed `subCost` directly;
  `byStem` indexes exact strings rather than equivalence-class representatives.
- `packages/sdk/src/seam-insertion.ts` delegates candidate spelling to this core
  lexicon and collects raw pronunciations by exact normalized-string equality.
  It contains no separate soft-match equivalence representation.

The reference's target forms are connected `وَءِننننَ` and isolated `وَءِننن`.
Held-duration collapse shortens these noon runs without changing their identity.

## Runtime validation and limits

A small local audit executed the actual `subCost` implementation with Node
22.22.2's TypeScript stripping; it required no dependencies or acoustic compute.
It enumerated the directed soft-match matrix over all 42 inventory phones and
computed a hypothetical closure solely for comparison. Exactly two additional
ordered pairs would enter that closure: ث→ص and ص→ث. Assertions passed that both
currently cost `UNIT` (10), while the direct ث→س and س→ص pairs still cost 5.
Every unequal noon pair costs 10, including noon→lam, noon→ya, and noon→meem.
The diagnostic is saved in `.autolab/source-soft-match-audit.json`.

The required representation defect is absent: the running implementation already
requires a direct pair, and the only hypothetical closure-only pairs do not
involve noon. There is no engine change to implement within this card's mechanism.
No engine or unit-test changes are retained, and no practice score has been run.
Targets won: 0; an overall score change has not been measured.

Modal runs 1 and 2 both failed through the authorized supervisor with
`NotFoundError: No such file or directory`, without producing a target trace.
Run 1 requested the supplied trace CLI on `realuser-v1-sob-049` at 36.44 s.
Run 2 requested a diagnostic using the card's listener evidence and the ordinary
engine and true-position replays, with a reporting hook in an ignored diagnostic
copy of the checker. Neither failure establishes a target verdict, and neither
is a practice score. No alternate Modal access or full test run was attempted.

The source audit and executable cost-matrix audit suffice to reject this specific
mechanism without a target replay: none of the actual lexical comparisons can
admit an edge by transitive closure. The target's exact accepted rival and
confirmation margin remain unverified in this run. Diagnostics remain under
ignored `.autolab/`; they do not modify the engine, scorer, reference assets,
supplied evidence, manifests, or experiment card.

This experiment tests matching semantics, not whether the recorded noon was
actually pronounced or how to resolve its acoustic confusion. Text and trace
inspection are not an independent listening judgment.

The killed outcome is recorded in untracked `.autolab/outcome.json`. No fold or
discard command was run.
