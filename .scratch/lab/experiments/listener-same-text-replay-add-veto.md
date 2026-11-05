# Same-text clean replay veto for ownership-free seam spellings

Status: premise cleared; corpus-wide construction and scoring pending. Experiment card unchanged.

## Prediction

Listener −3, overall −2 on the practice set, using the shipped listener. The
required continuation criterion is at least three target wins; the five targets
are those in the experiment card. No practice score has been requested.

## Mechanism

Re-land the final ownership-free enumeration from
`exp/listener-excise-certify-seam-waqf-adds`: whole pause-bounded utterances at ayah
and marked inner-waqf seams, stopped nasal retention, exact whole-word spelling,
neighbour/prefix/back-run/phrase and neighbouring-text-fragment exclusions, then
closed read-versus-omit neighbour certification after excising the candidate's
frames. The main phoneme string and tracker stay fixed.

Build clean reference contexts from the same EveryAyah source used by the earlier
same-ayah reference bank. This source is excluded from the evaluation reader pool
by the source registry (`docs/benchmarks/lab-reciters-v1.md`). Keep Arabic Hafs
murattal collections, one bitrate per source name, and the established seeded
three-collection sample. The preliminary probe joins preceding, current and following ayahs; the
shipped asset uses complete surahs, joined from the same source clips, before the
shipped phone transform. Its noise and simulated rooms come from public training
sources, rather than the practice augmentation bank. One augmentation is fixed
per collection/context before any result is examined.

The shipped SDK listener records its ordinary context-free-gap evidence. Replay
that evidence through the unchanged free tracker and checker, observing the
ownership-free enumeration before the new veto and neighbour certification, also retaining ordinary seam
spellings. A through-reading at an inner seam remains a replay when no pause is
present. An inspected seam counts as coverage even if no word is spelled. The offline
negative set joins all spellings from each distinct collection at each seam;
repeated streaming callbacks and overlapping contexts do not add replays.

Text of the two adjacent ayahs plus the seam's offset identifies a seam, without
session, reader or verse conditions in engine code. A spelling may proceed only
with at least two replays and no equal canonical spelling in that seam's clean
negative set. One matching negative suffices. Missing or incompatible coverage
abstains. All other warning paths retain their existing behaviour.

## Validation so far

- Read AGENTS.md, the card, the parent diff and recorded result, and the previous
  reference-bank construction and source-exclusion documentation.
- The supplied data mount contains practice, models and help-train, but no
  existing same-ayah reference bank. The generic corpus builder reconstructs
  contexts from the established public source.
- Initial Modal builds and six-shard suites passed: 747 core tests, 16 SDK tests,
  one SDK E2E test skipped.
- Modal run 4 passed both builds and 18 focused tests. Its ungated upper-bound
  replay confirmed all five planted identities in both the free engine and
  true-position arms: p19 at 106.36 s, p50 at 153.20 s, 4:157:24 at 59.40 s,
  9:5:15 at 116.44 s, and p49 at 158.48 s. These are not wins for the final veto.
- Run 4's corpus preparation failed because the node image lacked NumPy. Run 6
  installed the preparation dependencies and successfully prepared all 15
  contexts. The fixed collections are Juhaynee, Maher Al-Muaiqly and Minshawy
  murattal, the same seeded sample used by the earlier bank.
- Added a regression for clean through-reading coverage without an inner pause.
  Modal run 8 passed both builds and all 19 focused tests.
- Run 9 inspected all 15 completed reference records. P19, p50 and p49 each have
  two usable clean collections; 9:5:15 has three. Their negative spelling sets
  are empty. 4:157:24 has one usable collection in the preliminary protocol.
  This clears the acoustic/coverage premise for four targets, rather than
  claiming an overall gain. The training augmentation bank is 212 noise clips
  and 64 simulated rooms.
- Run 7's combined guarded engine replay exceeded the supervisor's 30-minute
  sandbox deadline; the completed clean reference records remain available.
  This is an execution failure, not a measured overall result or a premise kill.
  Run 12 retries the diagnostic without raw enumeration logging and skips
  sessions where every requested candidate fails coverage/existence by construction.
- Run 12 completed successfully (200.7 seconds). The guarded free engine and
  true-position arms both confirm p19, p50, p66 and p49. P31 abstains with one
  preliminary replay. Four target wins clear the three-target premise.
- Modal run 10 prepared every surah of all three source collections: 342
  recordings, all 20 shards exited zero, no unavailable clips reported.
  Full-surah contexts avoid the startup coverage loss in short crops. Run 11
  is constructing the corpus-wide asset, including the through-reading coverage
  and ordinary seam-spelling observations. No target-only asset will ship.

- Run 13 counted 111/342 completed full-surah records at inspection time.
  The input is 69.998 hours, with a longest recording of 136.84 minutes.
  Construction is resumable and the packer rejects incomplete or mixed-version
  banks. Run 14 waits for completeness, then packs and checks the shipped default
  asset rather than a diagnostic override.

## Limits

Canonical source recitations serve as clean negatives; this is an automated
text/acoustic probe, not an independent listening review. The fixed sample contains three distinct named reciters. The
phone formula matches the shipped transform, while its training noise and room
samples differ from the held-out practice augmentation bank. Missing tracker
pairing reduces usable replay coverage; a clean inner through-reading still counts. Target
selection is confined to a temporary externally supplied diagnostic request;
a target-only asset is not a shipping asset.
