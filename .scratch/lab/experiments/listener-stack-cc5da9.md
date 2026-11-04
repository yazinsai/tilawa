# Final-slot manner rekey with committed tracker ayah coverage

Status: implemented and practice-scored once; the predicted overall −3 was measured. Leave the merged change for the supervisor's fold gate.

## Prediction

On lab-v1.5 practice with the card's existing listener, listener −2, tracker −1 and overall −3. The three target incidents are the missed ayah-final wrong words at p78 `11:11:10` and p50 `37:84:5`, and the false skipped-passage warning at p46 `54:36:1`.

## Mechanism

This experiment combines `exp/listener-final-slot-manner-rekey` and `exp/tracker-stepped-ayah-not-skipped` without adding a mechanism or changing thresholds.

The manner rule operates on an already-proposed wrong word or omission at an ayah-final slot before a labelled pause. A mismatch between the last heard consonant's manner and the written pausal final replaces the lexical doubling and confidence cut, while retaining posterior support and the weak-evidence gate. A proposed omission becomes a wrong word only when its anchored slot contains at least two non-blank tokens including a consonant and has that manner mismatch.

The tracker exports an ayah after two consecutive zero-edit next-word entries within it, from the local path actually taken. The checker keeps that ayah off its whole-ayah absence ledger. This does not change position event timing or the word-level accusation rules.

## Merge and validation

Both parent branches were merged in the requested order. The listener branch conflicted with current main at the checker's final issue-processing block. The resolution retains the current formula-slot candidates, connected-madd accusations and bound-word return value, followed by the parent's unchanged manner processing. The tracker checker additions merged automatically.

The manner processing block, tracker source files and inherited unit tests were checked against their parent branches byte for byte. `lab-experiment.json` matches the original stack card byte for byte. `git diff --check` passes. No scorer, lab tooling, test data, evidence, manifest or listener model was changed.

The inherited tests cover manner mismatch and same-manner finals, pause boundaries, confidence and posterior support, omission reclassification versus an empty slot, clean ayah endings, consecutive local entries versus jumps and edits, and the coverage note's effect on the ledger.

Modal request 1 ran `bash tools/autolab_tests.sh` on six shards. Core and SDK builds succeeded. All six shard statuses were zero: 735 core tests and 16 SDK tests passed, with the existing SDK help end-to-end test skipped. The checker shard reported the known Vitest `onTaskUpdate` worker timeout after all its tests passed; the existing validation script accepted that specific infrastructure error. Logs are in `.autolab/modal-out-1/<shard>/log.txt`.

## Practice result

The single requested `practice-score .` measured commit `59c1ae38`. The supervisor returned `verdict: better`, with the expected overall −3:

| Practice phone | Main | Stack | Change |
| --- | ---: | ---: | ---: |
| Overall wrong calls | 33 | 30 | −3 |
| Missed mistakes | 22 | 20 | −2 |
| False warnings | 11 | 10 | −1 |
| Listener attribution | 32 | 30 | −2 |
| Tracker attribution | 1 | 0 | −1 |
| Checker attribution | 0 | 0 | 0 |

The result is recorded in `.autolab/practice-result-1.json`. No additional practice score was requested, and the engine was not changed after scoring. The aggregate counts match the stack prediction; the response does not contain individual incident keys or the fold gate's collateral and spread checks, so it does not independently establish that the three named targets were the only changes or that no incidents were lost. The parent's target traces remain the case-level evidence for the two mechanisms.

## Limits

The combined practice result establishes a net improvement of three wrong calls on current main. Coverage certifies local word entries rather than every phoneme of an ayah; manner rekey retains the parent's acoustic and pause restrictions. Reader robustness, incident losses, new false warnings on fully clean sessions and studio audio remain for the supervisor's incident comparison and fold gate. No fold or discard was performed here.
