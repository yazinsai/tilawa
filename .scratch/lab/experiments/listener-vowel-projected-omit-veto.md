# Vowel-projected single-word omission veto

Status: implemented; Modal validation blocked without command output. The
experiment card is unchanged. No premise kill or practice score is recorded.

## Prediction

The card predicts one of three omission false warnings removed, no lost calls,
listener −1 and overall −1 on practice. This is below the fold gate by itself.
The card's kill criterion requires two targets to prefer reading by
`PHRASE_SWITCH`; the task's final instruction requires three built target wins
before a practice score or a retained implementation.

## Mechanism

At an already-proposed one-word omission, take the onset of the counting visit
two nonempty words before it through the end of the counting visit two words
after it, staying within one surah and allowing an ayah seam. Pick the nearest
outer visits in temporal order around the proposal. Include the five written
words in order, grouped by their actual ayahs.

Reconstruct each stored student's top-k posterior row with the existing
`PosteriorFrames.row` contract. Sum probabilities into six separate vowel
classes (a, i, u, ā, ī, ū) and one other class containing every consonant, ghunna,
pause symbol and blank. Use other as the projected CTC blank. Remove all
nonvowels from each written word's labels. Reuse the ledger's forward sum with
word and ayah contiguous repeat arcs, comparing the intact span with the same
span missing the flagged word's vowels. Veto only when the intact path wins by
at least the existing `PHRASE_SWITCH` of four nats. Missing frames, a vowel-free
flagged word and nonfinite comparisons abstain. The candidate never edits the
primary evidence, tracker or alignment steps and adds no numeric threshold.

## Validation and limits

The target replay and supplied build/test harness were each requested twice
through the authorized supervisor on 2026-11-06. Runs 1 and 4 requested the two
target sessions, with both the actual free tracker/checker path and the
oracle-position diagnostic path. Runs 2 and 3 requested the six-shard
`tools/autolab_tests.sh` harness. All four returned only:

```
NotFoundError: No such file or directory
```

Each result reported zero spend and returned no command logs or shard output.
The first replay request used `git show` to select the baseline; the retry
used a saved baseline source file instead. The unchanged test harness failed
identically. Execution cannot be established from those results; they provide
no evidence about vowel presence or build/test status. Run 5, a minimal `pwd`
request to isolate the runner failure, returned the same error, zero spend and
no command output. The user was asked for the missing path or a
runner restore; no alternative Modal access was attempted.

Targets evaluated: zero. Targets won and margins: unknown. Tests passed: unknown.
Practice scores used: zero of three. No listener or overall change can be
reported. In particular, it would be incorrect to report zero target wins as a
measured kill. Resume validation before deciding whether the task's three-win
minimum or the card's two-target premise threshold is met.

### Retry after supervisor storage feedback

The supervisor's subsequent feedback reported `No space left on device` while
creating this experiment's Modal `runs` directories, followed by missing shard
status and log files. On 2026-11-06 the authorized runner was retried:

- Run 6: six shards of `bash tools/autolab_tests.sh`, with a 30-minute timeout.
- Run 7: two shards replaying p50 and p65, with a 20-minute timeout. Each shard
  was requested to run the baseline and instrumented candidate through both `runEngine` and
  `runTruePosition`, using the card's listener and stored phone evidence. The
  saved baseline was checked locally against the experiment's base checker and
  was identical. Diagnostic sources were placed under `.scratch/lab/vowel-probe/`
  for the supervisor snapshot.

Both retries returned `NotFoundError: No such file or directory`, zero reported
spend and no shard logs or event outputs. The new results do not identify the
missing path and do not establish that the command bodies ran. The earlier
storage feedback is an infrastructure finding, not a build/test or acoustic
finding. No additional practice score was requested and no premise kill is
claimed. The user was asked whether supervisor storage had been restored and
whether a missing-path diagnostic was available. The temporary snapshot sources
were removed locally after the requests completed; the ignored diagnostic
copies remain under `.autolab/vowel-probe/` for a future authorized retry.

The existing engine implementation and eight tests remain committed and
unchanged. Local `git diff --check` passed again. Target margins, target wins,
core/SDK build and test results, and all population safety claims still require
successful supervisor execution.

The proposed replays use the card's declared listener
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0` on phone audio. Labels and
target references select reported cases only, never engine behavior. These are
diagnostic replays, not practice scores or independent listening judgments.

Eight unit tests were added but could not be executed. They cover probability
aggregation, unlisted-mass reconstruction, six
separate vowel identities, changed consonants, a spliced-out vowel sequence,
word repeats, absent frames, a vowel-free target and impossible hypotheses.
Checker tests also exercise a proposed omission with and without vowel evidence,
and verify that the supplied evidence is preserved. Existing core/SDK tests
also remain unverified here. Local `git diff --check` passed.

Temporary diagnostic logging was removed from the engine. Diagnostic sources
were retained only under ignored `.autolab/vowel-probe/` artifacts. The source
candidate is left in place for restored supervisor validation. No scorer,
tooling, supplied manifest, test data, evidence or card was edited, and no fold
was performed. This write-up does not establish a population gain, lack of
losses, clean-session behavior or studio behavior.
