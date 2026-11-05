# Cross-reader posteriorgram word rank at seams

Status: needs-info. Acoustic premise passed on four targets; runtime integration
is blocked on the unavailable parent candidate-generator patch.
Experiment card unchanged.

## Prediction and mechanism

Card prediction: listener −2, overall −1 on lab-v1.5 practice, from two of four
seam additions, with about one loss. The later task instruction requires at least
three target wins before continuing; that instruction overrides the card's
two-target kill criterion.

Keep the channel-cost contiguous-greedy-span proposals and the excise-then-forward-sum
neighbour certification, replacing the absolute clean-seam log-odds cut with a
strict reference rank. Each query uses its original shipped-student frames,
including blanks, and subsequence DTW costs negative log posterior inner products.
Every query frame participates; reference leading and trailing frames are free.
Distance is divided by the fixed query frame count. A candidate must beat every
neighbour, next-word prefix and contiguous back-run reference. Ties and absent
competitor coverage abstain. No distance threshold is fitted.

## Premise check

The parent's committed result (`934515c5`) retains its findings rather than an
executable candidate-search patch. It reports that p19's emitted span extended
into the next word and named a different candidate, while its planted ومن span
was below the discarded absolute cut. It also reports that p49 failed word-1
neighbour certification on the seam holding its insertion. Those qualifications
are kept separate from the acoustic rank test.

The early driver therefore gives the acoustic premise an optimistic test: it
reads the current card's saved `exp-listener-stack-be988c-6a17f391` phone evidence
and selects the planted identities on their contiguous shipped greedy spans.
Case selection is supplied externally; no session, reader or verse condition is
added to the engine. P19's one-, two- and three-nasal contiguous endpoints are
included because held-run collapse makes each reachable to the span generator.
The other three queries include the complete greedy word, with its original
half-open frame range.

| target | query frames, 40 ms | shipped phones | planted identity |
| --- | --- | --- | --- |
| p19 `51:33:1` | 2552–2563 / 2566 / 2570 | `وَمِں` / `وَمِںں` / `وَمِںںں` | ومن |
| p33 `26:208:1` | 3904–3924 | `فَرِۦۦم` | كريم |
| p50 `37:99:1` | 3662–3693 | `وَقِفُۥۥهُم` | وقفوهم |
| p49 `26:123:1` | 3806–3827 | `كَرِۦۦم` | كريم |

The p49 free-engine trace (supervisor run 5) hears the complete word but produces
no added-word event there; it confirms an omission at `26:121:1` nearby. Acoustic
rank alone therefore cannot establish a p49 engine win.

## Reference construction and validation

EveryAyah is a training-source corpus excluded wholesale from the held-out
mp3quran pool by the repository's source registry. The early bank uses a seeded
three-collection sample of Arabic Hafs recitation, with translations, other
riwayat and tutor tracks filtered by source labels and one bitrate per named
collection. Collection aliases are not proof of speaker independence.

The request set includes every occurrence of the three planted word forms and
the neighbouring word forms, identified by generic mushaf text. Diacritics are
removed from the word identity key; the continuous canonical phones still align
each occurrence in its own context. Adjacent ayahs are joined before degradation
and decode. The shipped `degrade` and `phone_params` formulas use public training
noise shard 0 and 64 seeded simulated room responses, never the practice noise
or room bank. One augmentation is fixed per collection/context before ranking.
The augmentation bank contains 212 noise clips and 64 room responses.

The SDK's shipped streaming listener decodes the references with top-five
posteriors. The existing soft edit costs align contextual greedy phones; confident
soft matches at the existing 0.8 cut supply the neighbouring anchors. Word spans
run between those anchors. Prefix spans end immediately before the next label
on a closed CTC Viterbi reading of the reference word, without an extra confidence
cut; intervening blanks remain in the prefix. This construction was fixed before
any rank results were inspected. Back-run spans retain their contextual posterior
rows. Queries and references both use
`PosteriorFrames` to expand the same top-five representation.

Supervisor run 2 passed the baseline builds and six-shard suites: 734 core tests
and 16 SDK tests, with one existing SDK help E2E test skipped. Its known Vitest
worker RPC timeout was accepted by the supplied harness after every test passed.
Run 6 passed both prototype builds and three DTW unit tests covering complete-query
matching, free reference edges, held sounds, blank-frame costs, ties and missing
coverage. Python syntax and `git diff --check` passed locally.

## Result and limits

Supervisor run 8 finished all 20 shards successfully: 3,858 contextual decodes
covering 1,286 requested ayahs and 1,559 word/run occurrence requests. The seeded
collections were Juhaynee, Maher and Minshawy murattal. Run 12 verified completion,
built both packages, passed all four DTW/prefix unit tests, and ranked the fixed
queries using that complete bank. The student manifest SHA-256 was
`714e98addee73b1b8781ad7d99b6d3c745acd9116638baf4e12891f53a15af1b`.

Every planted identity ranked strictly nearer than every tested competitor, with
nonempty coverage of every required neighbour, prefix and back-run group.

| target | candidate references | competitor references | best candidate distance | nearest competitor distance |
| --- | ---: | ---: | ---: | ---: |
| p19, complete three-nasal span | 733 | 514 | 0.26158 | 0.67735 |
| p33 | 40 | 9,009 | 0.24525 | 0.77409 |
| p50 | 2 | 1,669 | 0.33168 | 0.88300 |
| p49 | 40 | 377 | 0.29465 | 0.67327 |

P19's shorter one- and two-nasal endpoints also ranked first. The complete rank
and coverage report is saved in
`listener-cross-reader-word-rank-adds-results.json`. Its `confirms` field means
the acoustic rank and competitor coverage passed; it does **not** mean that the
engine emitted or confirmed a warning. A partial-bank rank attempt failed with
a Modal volume resource error before producing any rank results.

The original parent candidate-generator patch is absent from its branch history
and reflog: only its card and discarded-result write-up were committed. No
matching patch was found within this worktree. A recovery audit of 173
unreachable commits and 9,294 source-sized blobs found four unrelated checker
variants and no seam-generator patch. Reconstructing that code from the
write-up would not establish the requested byte identity. The user was asked for
the saved patch while independent asset preparation continued.

The branch adds the independent DTW, strict rank and closed-prefix primitives,
the lossless packed-reference decoder, their unit tests, and an offline reference
builder/analysis driver. The driver no longer reads practice labels: its word and
span requests are supplied externally. These utilities are not wired into warning
emission. The main listener string, tracker and checker behaviour remain unchanged.

Supervisor run 13 prepares a generic all-word corpus with the same fixed readers
and augmentation, storing packed contextual decodes in the experiment's scratch
directory. It is the first of two planned batches and is not a shipped SDK asset.
The later batch and runtime integration depend on recovering the parent source.

Supervisor run 14 was rejected before executing the full suites with
`ResourceExhaustedError: too many layers in volume`; run 15 retries the supplied
six-shard test command and passed both builds and all six shards: 740 core tests
and 16 SDK tests, with the existing help E2E test skipped. The supplied harness
accepted the same known Vitest worker RPC deadline seen at baseline after every
test passed. Both new reference test files ran successfully.

A small Node source-level check also passed exact packed/live
posterior equality, malformed asset rejection, full-query DTW, strict rank
including non-finite rivals, and prefix cuts. It does not replace the full suites.

No practice score has been requested. This early bank and optimistic acoustic
query selection cannot by
themselves establish production candidate coverage, warning timing, clean-session
precision, reader robustness or studio behaviour. No independent listening
judgment or benchmark label correction is made.

## Continuation audit

The continuation confirmed that the parent branch still contains only its card
and result write-up. Preserving its generator and neighbour checks byte-for-byte
requires the original patch; reconstructing them requires clarification of that
constraint. The source request remains pending, and no warning-path change or
additional practice score has been made. This is a source dependency, not a fired
acoustic kill criterion; no killed outcome is claimed.

Supervisor result requests for run 13 and the scratch-corpus status inspection
(run 16) have not returned. Full-corpus completion and asset coverage therefore
remain unverified; the completed requested-word bank above is the only measured
reference coverage.
