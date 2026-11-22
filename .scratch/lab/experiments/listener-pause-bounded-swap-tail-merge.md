# Pause-bounded merge of a split substitution's tail

Status: implemented and validated; target won. Left for the supervisor's score and fold gate.

## Preregistered prediction

Listener −1, overall −1 on lab-v1.5 practice. The single target is
`lab-v1.5-p60/phone/warning:added_word/7:198:7/196.36`.
This is the card's one-win rider candidate, below the standalone fold gate.
The experiment card is unchanged.

## Mechanism

A long substitute can be split into a wrong-word head and an added-word tail.
Frame contiguity is too strict: CTC blank frames between phones need not mark
a word boundary. This change uses the checker's existing labelled-pause detector
and counting visits instead of adding a frame-gap constant.

After the final alignment and lexical candidates are judged, a support-path
addition can merge into the wrong-word accusation on the immediately preceding
written word only when:

- Its heard span follows the wrong-word span without overlap.
- No labelled pause or counting visit of any word separates the spans.
- The whole union, using the existing phrase-junk and edge-waqf stripping,
  passes `lexical.surplusWord` unchanged: a whole attested non-neighbour form,
  equal under soft costs only, without missing or added phones after held-phone
  collapse.

Spell-rule additions are exempt. A prior uncertain wrong-word decision cannot
own a tail. The wrong word retains the union as its reported heard span, while
its original confidence, support and confirmation timing still decide the
accusation. Thus a confident tail cannot promote an uncertain wrong word into
a new warning. The rule removes additions; it changes no listener decode,
tracker input, numerical constant, scorer, evidence, manifest or test data.

## Validation

Baseline true-position trace (Modal run 1, shipped `spliced-clean-int8` evidence):
the planted substitution is on 7:198:6, not slot 7. The wrong-word warning hears
`يَتتَبِعُۥۥ`; the next slot separately hears `كُم`. Both confirm at 198.96 s.
The label is the full substitution `يَتتَبِعُۥۥكُم` from 193.745 to 195.345 s.
The trace supports the split-word explanation; this is automated trace review,
not an independent listening judgment or a label change.

Boundary diagnostics and the changed true-position trace (Modal run 6) directly
validate all three premises:

| property | observed |
| --- | --- |
| wrong-word span | observations [181, 192), last frame 4871 |
| added tail | observations [192, 195), first frame 4875 |
| separating blank | frames 4872–4874, 3 × 40 ms = 120 ms |
| labelled pauses between spans | none |
| counting visits between spans | none |
| folded lexical result | `يَتتَبِعُۥكُم` |
| reported union | `يَتتَبِعُۥۥكُم` |

The observation spans are consecutive even though the frame spans are not.
At the observed 40 ms stride the existing pause detector requires 8 blank frames.
No new adjacency duration is introduced.

The changed trace has no suspected or confirmed added-word event at the target.
The wrong word still confirms at 198.96 s, with confidence 0.9294454545, now
reporting the whole substitute. The result holds under shipped policy and the
trace's three diagnostic policies with confidence and/or local support disabled.
**Targets won: 1.** This is a trace result, not a measured overall practice delta.

Seven direct unit tests exercise the merge with a small artificial vocabulary
through the actual folded lexical test: a short blank, labelled pauses, all
intervening boundaries, counting visits including overlapping visits, boundary
visits that do not separate the spans, whole-word/soft-cost/neighbor rejection,
and the existing-accusation and spell-rule exemptions.

Attempted synthetic streaming fixtures aligned the entire substitute as one
wrong-word span even without this merge; they could not establish a regression
for split accusations and were removed. The target trace supplies the integration
check. All existing tests passed in Modal run 5 across six shards; its only failed
cases were those discarded synthetic tests. Modal run 8 reran the final affected
core/SDK shard and both builds successfully (116 core tests and 5 SDK tests).
Across the shards, the final suite has **722 core tests passed, 16 SDK tests
passed and 1 existing SDK test skipped**. The existing checker test shard hit
Vitest's worker RPC deadline after every test passed; the repository's unchanged
validation wrapper accepts that specific condition.

## Limits

This checks one target with shipped evidence. It does not measure an overall
practice delta, reader concentration, clean-session losses or studio outcomes;
those belong to the supervisor's score and fold gate. No practice-score run
has been requested by this builder.

An independent addition whose union with a preceding wrong word happens to
spell one attested word can also be absorbed. A real split substitute with a
hard phone error, a labelled pause or an intervening counting visit stays
separate. The rule depends on the final alignment still retaining the wrong
word's span when the addition becomes visible.
