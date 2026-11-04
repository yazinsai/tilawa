# Excise and certify spelled words at ayah and waqf seams

Status: implemented; three targets confirmed; final practice overall +11, listener +5.
The premise holds, but the measured change fails the improvement gate. Patch left
in place for the supervisor's fold gate, as instructed.

## Prediction

Practice set `lab-v1.5`, shipped listener evidence, branch based on `0c696f5a`.
Three of five target additions won: listener −3, overall −3 (floor −2).
The experiment card is unchanged.

## Mechanism

At an ayah boundary or marked inner-waqf boundary, visits bound a window containing
the previous word, the next word and one following word. The visits need not be
complete or free of hard edits. The final rule enumerates whole greedy utterances
between detected blank-owned pauses, without assigning their phones to a visit
first. This prevents excising part of a word or part of a held run.

A held nasal after a candidate's surplus phones is retained as ن. The existing
whole-word lexicon test still requires an exact consonant spelling, with only
the existing soft vowel differences. Neighbours, their prefixes, nearby back-runs
and non-Quran phrases remain excluded. The final rule also excludes substrings
of the neighbouring text, including its pausal pronunciations and contiguous
runs. A candidate cannot join phones across a detected pause or phrase-labelled
speech.

For each spelling, its frames are cut from a read-only CTC calculation. The
remaining rows score `[previous, next, following]` against the same sequence with
each of the two neighbours omitted. Canonical and pausal forms are available to
both competitors, with the ledger's contiguous word-repeat arcs and no filler.
Both neighbours must win by the existing `PHRASE_SWITCH` (4 nats). This certifies
presence independently of the greedy assignment. The following word is context.

A passing candidate uses the folded lexical added-word confirmation. Existing
lexical confirmations and completed decisions retain their existing behaviour.
No model, main decoded string, tracker, scoring code, evidence or threshold changes.

## Validation

- Modal run 1: baseline true-position trace of the first inner-waqf target.
  At 4:157:24 the ordinary addition suspicion (`وَءِں`) changes to a wrong-word
  suspicion (`ںںفَاا`) and ends uncertain. Confidence is not its blocker.
- Modal run 2: TypeScript build and five new CTC/excision unit tests passed on
  all four shards. The target driver was absent from the snapshot under `.scratch`.
- Modal run 3: the transported diagnostic needed the existing evidence chunk
  converter, rather than appending saved-file posterior metadata directly.
  Corrected for the subsequent run; this was a diagnostic failure.
- Modal run 4: builds passed, but the full suite found 17 failures, including
  additional warnings on clean synthetic and tlog fixtures. The five forward-sum
  unit tests passed; both new positive integration cases passed. The inner-pause
  control initially also allowed the ordinary support path; that test was
  subsequently isolated with an infinite ordinary-support requirement.
- Modal run 5: full engine and true-position replays confirmed four targets.
  Each confirmed observation is the full planted spelling, rather than the
  fragment or tail reported by the earlier excision experiment.

Initial build, before narrowing:

| Target | Observation | Previous / next read-over-omit, nats | Engine / true position |
| --- | --- | ---: | --- |
| p19, 51:33:1 | وَمِن | 59.59 / 25.89 | confirmed / confirmed |
| p50, 37:99:1 | وَقِفُۥۥهُم | 43.51 / 31.94 | confirmed / confirmed |
| p31, 4:157:24 | وَءِن | 12.78 / 13.67 | confirmed / confirmed |
| p31, 4:170:12 | ءِن (fragment) | 21.79 / 10.85 | uncertain / uncertain |
| p66, 9:5:15 | وَءِن | 31.43 / 6.18 | confirmed / confirmed |

Margins in this table use the truth-timed previous-through-following window as
an independent diagnostic, not the engine's exact visit bounds. The separate
full engine replay supplies the confirmation result. At 4:170 the full فإن is
not returned by the existing non-neighbour spelling test; the emitted إن
fragment has confidence 0.57994 and stays uncertain.

The unit regressions show an important limitation: read beating complete omission
does not prove that all of a neighbour's phones survived. A substring can be a
mushaf word and leave enough of the neighbour to beat total omission. Practice
was requested before any narrowing, as required after four target wins.

Practice score 1 returned an execution error: a session shard exceeded Modal's
1800-second limit, so there is no overall delta from that attempt. The repeated
seam calculation was then cached by word, frame bounds, and the immutable greedy
phones plus their phrase labels. Cached span offsets are relative to the seam
window and expired when the window leaves history. This changes computation only;
no candidate, cut or decision criterion was narrowed before obtaining a score.

Modal run 8 rebuilt the cached version and replayed every target in both modes.
All ten target event records were exactly equal to run 5. The run finished in
148.4 seconds versus 414.9 seconds for run 5. Practice score 2 was then requested
as a retry of the same decisions.

The clean-fixture probe found that six of seven tlog warnings were substrings of
nearby written words. The seventh, `فَءِن` at a clean 100:11 seam, cut the first ن
of a held run and left another ن immediately after it. Its margins were 59.21 and
6.05 nats. A positive read-versus-omit margin therefore does not, on its own,
prevent taking part of a neighbour's spelling or part of a held run.

## First completed practice score

Attempt 2, with the equivalent cache: overall 36 → 1170 (+1134), listener
32 → 202 (+170), tracker 1 → 3 (+2), checker 3 → 965 (+962). Misses fell from
24 to 12, but false warnings rose from 12 to 1158. This decisively fails the gate.
Attempt 1 timed out and supplied no score.

Only after this completed score, the candidate set was narrowed to entire speech
utterances between two detected, blank-owned pauses. A cut cannot begin or end
inside a word or held run, or bridge an internal pause. The existing minimum
pause detector is unchanged. A spelling that is a substring of any nearby
written pronunciation or contiguous run is also excluded, including pausal
pronunciations. These are text and boundary guards, not a new acoustic threshold.

The read-only boundary probe had three full planted spellings with both bounding
pauses: p19, p50 and 4:157. The 9:5 spelling lacks the right bounding pause, so
this refinement no longer claims it.

## Final validation

Modal run 9: builds, 709 core tests and 16 SDK tests passed on six shards; the
optional SDK end-to-end test was skipped. The harness
accepted the documented Vitest worker RPC timeout after all tests passed. The
previous clean-fixture and lexical regressions are gone. New tests cover a
blurred neighbour at each seam type, both pause boundaries, a bare nasal, a
lexical substring of a neighbour, and the cut CTC recurrence against explicitly
concatenated rows, including repeat arcs.

Modal run 10: the narrowed engine and true-position replays both confirm p19
at 106.36 s (`وَمِن`), p50 at 153.24 s (`وَقِفُۥۥهُم`), and 4:157:24 at 59.40 s
(`وَءِن`). The other two targets do not confirm. Three of five retained; the
premise kill criterion does not fire. Temporary diagnostic scripts were removed.

## Final practice score

Attempt 3 scored committed engine version `6cf7f34d`:

| Attribution | Main | Experiment | Change |
| --- | ---: | ---: | ---: |
| Overall | 36 | 47 | +11 |
| Listener | 32 | 37 | +5 |
| Tracker | 1 | 1 | 0 |
| Checker | 3 | 9 | +6 |

Misses fell from 24 to 17, but false warnings rose from 12 to 30. Bounding the
candidate by pauses and excluding neighbouring text fragments removed most of
the broad version's false warnings, but did not make this rule precise enough.
The final result is worse and does not meet the predicted improvement or the
fold gate's overall/component requirements.

All three allowed practice requests were used (one timeout, two completed).
The aggregate receipt does not identify fully clean sessions or report studio
or leave-reader-out results; those gates have not been established here. Three
target confirmations remain, so this is not a premise kill. The tested engine
change remains committed for the supervisor's scoring and gate decision.

## Limits

The hypothesis is about exact shipped-token spellings. No confusion-channel
substitutions or second listener are consulted. A word whose consonants are not
spelled cannot pass. The checker must retain the seam and its following context
long enough to certify and confirm a candidate. Truth-timed diagnostic margins
alone do not establish an engine win.
