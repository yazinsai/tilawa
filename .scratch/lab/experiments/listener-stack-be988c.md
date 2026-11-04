# Stack final-slot manner rekey and pause-bounded swap-tail merge

Status: implemented; tests passed, one practice score pending.

## Prediction

Listener −3, overall −3 on lab-v1.5 practice. The two final-slot wrong-word
misses and the split-substitution false addition should all disappear:

- `lab-v1.5-p50/phone/mistake:wrong_word/37:84:5/50.27`
- `lab-v1.5-p60/phone/warning:added_word/7:198:7/196.36`
- `lab-v1.5-p78/phone/mistake:wrong_word/11:11:10/143.13`

## Mechanism and merge

This is the requested merge of `exp/listener-final-slot-manner-rekey`
(`cd54354e`) and `exp/listener-pause-bounded-swap-tail-merge` (`11cccac4`).
No additional rule or numerical change is introduced. The stack's original
`lab-experiment.json` is preserved byte for byte.

The manner rule applies to an existing wrong-word or omission accusation on an
ayah-final slot before a labelled pause. A contradiction between the last heard
consonant's manner and the written pausal final replaces lexical doubling and
the confidence cut, while retaining posterior support and weak-evidence gates.
An omission with at least two heard tokens including a consonant becomes a
wrong word on that same slot.

The tail rule merges a support-path addition into the preceding wrong-word
accusation only when no labelled pause or counting visit separates their spans
and the union passes the existing whole attested non-neighbour word spell test.
The original wrong-word evidence still controls confirmation. Spell-rule
additions and previously uncertain wrong-word decisions remain exempt.

The first checker conflict preserves the current formula and madd rules and
inserts the parent manner block before the existing judge return. The second
preserves that block and extends the return with the tail parent's counting
visits. All other parent changes, tests and write-ups merge unchanged.

## Validation

`bash tools/autolab_tests.sh` passed on all six Modal shards (run 1), including
the core/SDK builds: 734 core tests passed, 16 SDK tests passed and one existing
SDK test skipped. Both parent suites passed unchanged: 12 manner tests and
seven swap-tail tests. The checker shard hit the known Vitest worker RPC
deadline after every test passed; the repository's unchanged wrapper accepts
that specific condition.

A local source comparison verified that the manner table, confirmation helpers,
closing-slot block, tail-merge block, helper and unit tests match their parent
revisions exactly. The experiment card also matches the initial stack commit
byte for byte. No scorer, lab tooling, test data, evidence or manifest was edited.

One `practice-score .` is pending to measure the combined behavior against
current main.

## Limits

The parents' earlier independent gains do not establish this stack's current
practice result. Their original write-ups document their target trace evidence;
this experiment adds no independent listening judgment or label changes.

Manner contradictions can also arise from phone-channel blur. Tail merging can
absorb a separate addition whose union happens to spell one attested word.
Reader concentration, clean-session warnings, incident losses and studio audio
must be assessed from the practice score and the supervisor's fold gate.
