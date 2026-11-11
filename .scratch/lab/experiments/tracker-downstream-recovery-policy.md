# Downstream recovery policy

Status: needs-human — supervised Modal execution is unavailable. This is an
incomplete implementation checkpoint, not a trained policy or a measured kill.

## Prediction

Tracker −1, overall 0. Correcting local recovery may remove the false
`skipped_passage` at `lab-v1.5-p46/phone/54:36:1/121.84`, while exposing a listener
false accusation nearby. The card is unchanged.

## Mechanism and implementation checkpoint

The intended policy learns preferences over existing local recovery actions from
complete tracker/checker continuations, including downstream accusations. Its
training streams must use help-train training speakers' none and omission takes,
retain uncertainty in scripted labels, and exclude release-held-out speakers.
Speaker-grouped partitions must freeze training and model selection before any
practice evaluation. Further collection must include states visited by the learned
policy. Runtime features must describe relative candidate locations, costs and
history; they must contain no reader, session or verse identities. Checker rules
and available movement transitions must stay unchanged.

Code inspection established that `local.ts` retains an entire dynamic-programming
row of reference endpoints. It computes insertion, substitution, jump and deletion
transitions before selecting the minimum-cost endpoint. Selection can therefore
be intercepted without introducing a new movement edge.

The checkpoint adds an optional `TrackerOptions.recoveryAction` callback, passed
to local search with the evidence timestamp, previous local position, default
selected endpoint and finite retained actions. Each action exposes its endpoint,
run origin, edit count and accumulated cost. The callback may choose an existing
endpoint; an unavailable or non-integral choice is rejected. Omitting the callback
retains the existing selection. These raw reference offsets are for rollout
interventions, not learned features.

Unit tests cover preservation of the default positions and costs, selection of an
existing alternative with save/restore, and rejection of unavailable actions.
No learned policy, weights, training examples, or production action override has
been installed. The checkpoint alone is expected to preserve default behaviour.

## Validation attempted

Three requests used only the supplied supervisor command:

1. `modal-run` for storage/help-train inventory and the supplied trace command at
   the target incident.
2. `modal-run` for a baseline replay with action collection. The temporary local
   diagnostic replays the shipped tracker and checker using the existing chunk
   function and finalizes the session. Its optional rollout mode intervenes once
   at a retained endpoint and completes the same engine continuation; it reports
   final checker states, not a replacement practice score. That diagnostic was
   removed after the execution failure; no case-specific diagnostic code is
   retained in the branch.
3. `modal-run --shards 6 -- 'bash tools/autolab_tests.sh'` for build and core/SDK
   tests, including the added unit tests.

All three returned exactly `NotFoundError: No such file or directory`, without
per-shard output or a command log. The local result records are
`.autolab/modal-result-1.json`, `modal-result-2.json`, and `modal-result-3.json`.
Each reports zero dollars spent. The error does not establish which server-side
file or mounted resource is missing. No inference, replay, training, build or test
success is claimed. Local `git diff --check` passed.

## Limits and next required work

The target's action set and downstream outcomes remain unmeasured. In particular,
there is no evidence that every available restart action produces the same
whole-ayah skip through the existing decision horizon. The card's kill criterion
has not fired. Target wins and component/overall changes are unknown, not zero.
No practice-score call was used.

Restore supervised execution, run the action audit and tests, then either document
the measured premise kill and revert the hooks, or build the training streams,
collect complete downstream outcomes, train and aggregate policy-visited states,
freeze speaker-grouped model selection, and validate the resulting policy. The
three-target rule and practice-score requirement still apply. This checkpoint
must not be treated as a finished recovery-policy experiment or folded as one.

Scorer, lab tooling, datasets, evidence, manifests, experiment card, listener and
checker rules were not changed. No independent listening judgment was needed or
claimed: execution failed before incident triage could produce new evidence.
