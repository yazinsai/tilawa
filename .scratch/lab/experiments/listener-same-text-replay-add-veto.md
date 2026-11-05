# Same-text clean replay veto for ownership-free seam spellings

Status: implementation complete; shipped-asset validation pending. Card unchanged.

## Prediction and mechanism

Predict listener −3, overall −2 on practice with the shipped model. Continue to
practice scoring only if the completed mechanism wins at least three targets.

Re-land the final enumeration from `exp/listener-excise-certify-seam-waqf-adds`:
whole pause-bounded greedy utterances at ayah and marked inner-waqf seams, stopped
nasal retention, exact whole-word spelling, neighbour/prefix/back-run/phrase and
neighbouring-text-fragment exclusions. After cutting the candidate frames,
certify both neighbours with the existing closed read-versus-omit forward sum
and PHRASE_SWITCH. No numeric constants change.

The new veto identifies a seam by the full phoneme text of its adjacent ayahs
and its offset. It requires at least two distinct clean source collections. Any
matching canonical spelling from any clean replay vetoes the candidate; missing
or incompatible coverage abstains. Apply this after the parent enumeration
selects its candidate, so rejection cannot substitute another word. The main
listener string, tracker and other warning paths keep their existing behaviour.

## Clean replay asset

The supplied mount had no existing same-ayah reference bank. Reconstruct the
established EveryAyah source, which the evaluation reader registry excludes
wholesale (`docs/benchmarks/lab-reciters-v1.md`). Preserve the prior generic Hafs
murattal filters, one bitrate per collection and seeded three-collection sample:
Juhaynee, Maher Al-Muaiqly and Minshawy murattal. No result-dependent selection.

Join each source's ayah clips into complete surahs: 342 recordings, 69.998 hours.
Apply the unchanged shipped phone transform with deterministic per-context
seeds. Its training augmentation sources contain 212 public noise clips and 64
simulated rooms; they differ from the practice augmentation bank.

Record ordinary SDK listener evidence, including the folded context-free gaps.
Replay through the unchanged free tracker and checker with an empty veto asset.
Observe ownership-free spellings before certification/veto and ordinary seam
spellings. A usable inspected seam counts even if it spells nothing; an inner
through-reading without a pause is also a replay. Deduplicate callbacks and
source collections before unioning each seam's spellings.

The packer requires every corpus record and matching reference, inventory,
listener SHA, job identity and collector fingerprint; it rejects incomplete or
mixed banks. Corpus collector fingerprint:
`3445d2b3ca9111f2468c167a3b8c270190b9ee2615d7a0fb0306a868058903b7`.
Listener SHA:
`714e98addee73b1b8781ad7d99b6d3c745acd9116638baf4e12891f53a15af1b`.
Moving the consumer veto after candidate selection does not change the empty
asset collector's observations.

Builders: `modal/student_seam_replays.py` (`--all-seams --whole-surahs --prepare`),
`packages/sdk/src/seam-replay-build.ts` (`--jobs --model --out --resume`), and
`packages/sdk/src/seam-replay-pack.ts` (`--jobs --model --bank --out`). All acoustic
work runs through the authorized supervisor. The collector resumes matching
cached records, and the packer writes the declared engine data module.

## Validation

- Initial Modal builds and six-shard suites passed: 747 core tests and 16 SDK
  tests; one optional SDK E2E test skipped. The final suite is pending.
- Run 4's ungated enumeration confirmed all five planted words, in both the
  free-engine and true-position arms. This is an upper bound, not final wins.
- Runs 8 and 18 retain 19 passing focused tests for excision, pausal handling,
  neighbouring-text exclusion, coverage, version/inventory compatibility and
  matching-word vetoes. Run 18 also tried a two-candidate fixture whose positive
  control lost the neighbour fit; that invalid regression was removed.
- Preliminary three-ayah contexts (15 recordings) left four target spellings
  absent from at least two clean replays. Run 12's guarded free engine and
  true-position arms both confirmed p19, p50, p66 and p49. P31 abstained with one
  replay. This preliminary four-target result clears the acoustic premise.
- Run 10 prepared all 342 full-surah recordings without unavailable clips.
  Run 11 exceeded its sandbox deadline; run 15 resumed the matching records and
  completed the bank with all 20 shards exiting zero. Waiting run 14 also timed
  out without producing an asset. These are execution limits, not scores.
- Run 19 packages the complete asset, runs all six core/SDK test shards, and
  checks the five targets through the shipped default asset without an override.
- No practice score has been requested yet.

## Limits

Canonical source recitations supply clean negatives without an independent
listening review. Three named reciters and one fixed augmentation per recording
cannot cover every realization or microphone. Missing tracker pairing reduces
usable coverage; full-surah context reduces startup loss. Matching clean words
are a conservative existence veto, while unseen one-off blur remains possible.
Target requests belong only to a temporary diagnostic, which will be removed.
No test identities enter the shipped code or corpus selection.
