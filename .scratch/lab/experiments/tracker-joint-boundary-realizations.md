# Joint stop/continuation boundary realizations

Status: killed on the premise; no engine changes.

## Prediction

Tracker −1, overall 0. The proposed shared pronunciation state would remove a
jointly incompatible rhyme-restart path, recovering the tracker warning while
potentially exposing the documented nearby listener warning. Target:
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

## Mechanism and premise check

The card proposes binding the preceding word's ending and the following word's
onset to one stop-or-continuation state. Both complete alternatives would remain
when pause evidence does not resolve them. Phone-edit charges, movement edges,
commitment margins and checker handling must stay unchanged. Its kill criterion
is an equally priced, jointly consistent pronunciation path for the winning
restart.

The inspected engine has no independent pronunciation selection in this search.
`createTracker` starts and advances `createLocalSearch` in `local.ts`. That search
reads `index.phonemes` once and uses its characters in substitution, insertion
and deletion transitions. Its state is a reference offset, accumulated cost,
run origin and edit count; it has no pronunciation-variant state. Jump arcs
enter that same reference string. Tracker global and anchor fits also align
against fixed strings.

`loadReferenceIndex` concatenates each asset ayah's already composed `p` string.
It separately constructs `wordForms` and `isolatedWordForms`, including the plain
restart onset for assimilated words. The consumers of those alternatives are
checker modules, not the tracker or local matcher. Thus the restart cannot gain
from independently selecting those alternatives on adjacent word matches.

The stored forms at the disputed boundaries also coincide:

| Reference word | Form used by the tracker | Isolated form |
| --- | --- | --- |
| 54:36:4 | `فَتَمَاارَو` | `فَتَمَاارَو` |
| 54:36:5 | `بِننننُذُر` | `بِننننُذُر` |
| 54:37:1 | `وَلَقَدڇ` | `وَلَقَدڇ` |
| 54:37:9 | `وَنُذُر` | `وَنُذُر` |
| 54:38:1 | `وَلَقَدڇ` | `وَلَقَدڇ` |

In particular, the restart's rhyme-ending/next-onset sequence is already
`وَنُذُر | وَلَقَدڇ`. Sharing a boundary choice among these identical forms
does not change any reference symbol or edit price. Requiring a partially
matched source word to finish before a jump would instead change the existing
movement semantics, which this card explicitly preserves.

## Validation

Source and reference-asset audit completed on the experiment's starting commit
`9f4481f2`. No engine change was necessary to establish the missing premise;
therefore no behavior tests were added or modified.

Modal run 1 requested the supplied target trace. Run 2 requested
`bash tools/autolab_tests.sh` on six shards, covering the existing core/SDK builds
and test suites. Both returned `NotFoundError: No such file or directory` without
command logs; neither provides a test or replay result. Run 3 requested a
read-only audit guarding access to independent forms and comparing the tracker
events against an ordinary replay. It returned the same infrastructure error,
also without logs. Dynamic validation remains unavailable; no successful replay
or passing test suite is claimed.

The experiment is killed because the hypothesized independently selected
boundary realizations do not exist in the tracker, not because validation
infrastructure failed. Targets won: 0; no prototype or practice score run, so no
overall or component score delta is claimed. No scorer, tooling, card, reference
asset or evidence was edited.

## Limits

This tests the card's representation premise, not whether a richer pronunciation
graph would improve other recitations. The reference contains pause and connected
forms for other words, and the checker uses them. The finding does not establish
that every reference boundary or every acoustic alignment is correct. No audio
label changes or independent listening judgments are needed to establish which
reference representation the tracker reads.
