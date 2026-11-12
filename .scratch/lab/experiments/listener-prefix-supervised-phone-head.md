# Prefix-supervised phone-head refit

Status: awaiting the acoustic premise audit; supervisor compute is unavailable.
The experiment card is unchanged. No engine change, model export or practice
score has been made, and no kill criterion has been established.

## Prediction and mechanism

Predict listener −1 and overall −1 on the practice set, with no losses predicted.
Whole-recording CTC permits a phone occurrence to be emitted later than its
acoustic source. The proposed experiment freezes the shipped encoder and refits
its phone head with ordinary whole-recording CTC plus internal-prefix losses.
Prefix supervision must require completed occurrences while marginalizing the
partially observed final occurrence; it must not require the full transcript or
allow arbitrary progress through the remaining transcript.

Uncertain prefix targets must come from transcript-constrained training
alignments. Both the prefix arm and an ordinary-CTC control must use identical
authorized recordings and matched compute, retain deployed streaming context,
and select prefixes and checkpoints only within speaker-grouped training
partitions. Help-train use is restricted to training speakers' none-labelled
takes. Ordinary reading and real pauses remain preservation examples. Export
must preserve the inventory, 40 ms clock, tracker, checker and insertion
verification.

## Premise audit and validation

Before refitting, at least two registered targets must contain a missing planted
consonant whose transcript-conditioned emission mass is concentrated after the
addition's acoustic end, inside following speech. Weak emission or a borrowed
neighbour phoneme alone does not establish that premise.

An ignored diagnostic is prepared under `.autolab/`: it captures full-vocabulary
log probabilities directly from the normal SDK streaming backend, preserving
its persistent state and frontend, rather than reconstructing the rounded
top-five evidence. A local CTC forward-backward diagnostic keeps occurrence
identity, reports both first-entry and occupancy distributions, and compares
multiple surrounding transcript lengths. The intended report includes the
supplied addition and following-word intervals, greedy phones within the
addition, emission quantiles, mass after the acoustic end, mass after an
additional 160 ms boundary margin, and mass inside following speech.

The occurrence calculation passed exhaustive-path checks for four synthetic
transcripts, including repeated labels: total sequence probability, each
occurrence's occupancy, and each occurrence's first-entry distribution agreed
with enumeration of every five-frame, three-class CTC path. Python diagnostic
syntax checks also passed. These validate only the diagnostic calculation,
not the engine change or acoustic premise. No engine unit tests were changed
because no engine behavior has changed.

Supervisor requests 1 and 2 (2026-11-12 UTC) both returned
`NotFoundError: No such file or directory`, with zero compute charged and no
command output or downloaded artifacts. Request 1 attempted a read-only data
and dependency inventory; request 2 attempted the same inspection from an
ignored script. A minimal third request (`pwd`) is pending to distinguish
failure to start the remote environment from a data-inspection failure.

## Limits and next step

There are no target measurements, trained checkpoints, exported bundles or
practice deltas. The supplied label boundaries have not been independently
reviewed here; transcript-conditioned alignment is not independent listening
evidence. No authorized independent audio service is connected. Infrastructure
failure cannot be interpreted as zero target wins or as a falsified mechanism.

Restore the supervisor environment, finish the premise audit, and only then
decide whether the refit is warranted. If fewer than two targets meet the
premise, record a killed outcome without training. If the premise holds, build
and test prefix marginalization, train both matched arms, verify a complete
primary-model bundle, and check target recovery. A built change recovering
fewer than three registered targets must be reverted; at least three recoveries
require an official practice score before deciding. All three allowed practice
scores remain available. No fold or discard was performed.
