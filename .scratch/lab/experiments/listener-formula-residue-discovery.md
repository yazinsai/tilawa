# Formula residue discovery

Status: blocked on supervisor execution; premise and target gain unmeasured.

## Prediction and mechanism

The card predicts one recovered added word, with listener −1 and overall −1.
A permitted formula and an extra word may share an existing pause-bounded
utterance. Whole-utterance word spelling then fails even when the extra word
would spell successfully on its own.

The proposed discovery grammar is `formula* word formula*`, with at least one
formula and at most one complete lexical word. Every segment must consume a
nonempty, disjoint token interval, and the intervals must cover the utterance.
Only existing formulas and spelling costs may participate. Formula segments
retain their exemption; the residual word must pass the existing neighbour,
prefix, back-run and local-substring exclusions, posterior neighbour
certification and original-audio teacher insertion gate. Pause boundaries,
primary decoding, tracker input and existing candidates must remain unchanged.

## Source audit

The current SDK discovers candidates through a read-only tracker/checker replay
in `packages/sdk/src/seam-insertion.ts`. The pause-bounded utterance enumerator
is in `packages/core/src/checker.ts`. It rejects an utterance containing any
formula-labelled token, then requires the entire remaining utterance to pass
`lexicalWords(index).surplusWord`. A candidate must also pass the local-fragment
exclusion and neighbour certification after excision. The SDK subsequently
verifies its exact word spelling on original audio with the existing teacher
insertion lattice and carries the resulting marginal as auxiliary metadata.

This source audit identifies a possible entry point for the proposed parse; it
does not establish that any registered target contains the required material.

An uncommitted diagnostic was prepared to enumerate the existing native-clock
pause-bounded utterances around all six targets and search disjoint formula
prefixes/suffixes around one residual word. It uses the existing word spelling
and exclusions. It also searches a generous closed-alignment bound using the
same formula edit costs, to avoid confusing the substring aligner's selected
path with absence of any full-span fit. Positive controls cover both orders and
formulas on both sides. The diagnostic has not executed; neither its controls
nor any target counts are validated. It is retained in ignored
`.autolab/formula-residue-audit.ts`, outside the committed engine code.

## Validation and limits

The authorized supervisor runner failed before producing container output:

| Run | Request | Result |
| --- | --- | --- |
| 1 | Inspect practice data paths | `NotFoundError: No such file or directory` |
| 2 | Supplied six-shard core/SDK build and test harness | `NotFoundError: No such file or directory` |
| 3 | Minimal `pwd` execution check | `NotFoundError: No such file or directory` |
| 4 | Formula/residual target audit | Pending when this report was drafted |

The minimal command failure establishes that the error is not evidence against
the parse mechanism. The runner provided no more specific missing path. A
request to restore the authorized execution route was sent to the user.

No engine behaviour or unit tests were changed. The experiment card, scorer,
lab tooling, supplied data, manifests and evidence remain untouched. No new
model bundle is required by the proposed mechanism. No practice score was
requested, so all three scoring allowances remain available. No fold was
performed.

There is no measured target-win count, listener delta or overall delta, and no
basis for declaring the experiment killed. Resume by restoring supervisor
execution and running the premise audit. Only if that premise survives should
the generic discovery hook and behavioural tests be implemented, followed by
original-audio target verification and the task's three-target pre-score rule.
