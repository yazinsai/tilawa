# Checker: acoustic ownership beside a gap

Status: experiment, awaiting practice measurement and the user's decision

## Preregistered prediction

The card predicts **2 fewer missed mistakes and 2 fewer false warnings** (4 fewer wrong calls overall), with unchanged checker attribution under oracle inputs. Both old and new practice sessions, clean sessions, and concentration by reader and passage must be checked. A change of 2 or fewer is noise. No fold or discard is authorized.

The development pattern is a short omitted word whose sounds resemble the noisy word beside it. The alignment lends that word's sounds to the omission, then places the gap one word away. Practice examples motivating the hypothesis were p79 (20:25:4/5), p14 (44:5:4/5), and p26's skipped passage. These are development evidence, not a held-out validation claim.

## Change

For an existing gap, consider moving the adjacent spoken visit to the other side of that gap. Compare the complete canonical spoken sequences over **identical posterior frames**, preserving a correct spoken word at each outer edge. Require the alternative to win by the existing `SUPPORT` margin. The raw edit-path cost cannot resolve this ambiguity: it already chose its best path. Posterior sequence likelihood supplies independent evidence about which whole word was spoken.

Only a single adjacent visit can move. The guard checks the actual selected visits, chronological order, monotonically forward reference consumption, and exact ordered consumption of every observed phoneme in the replaced interval. Junk, overlapping added speech, missing posterior coverage, and repeated/backward paths abstain. The repaired path feeds the normal checker gates. No confirmation thresholds change, and evidence without posteriors is unchanged.

This differs from the earlier `checker-skip-boundaries` experiment: it supports single omitted words, uses complete acoustic hypotheses instead of selecting a gap boundary from edit fit, and repairs the alignment path so downstream evidence gates see the same ownership. Existing joint-addition recovery is retained.

## Development checks

The new synthetic regression uses an independent passage (52:26): omit إنا, then hear the first consonant of كنا as ء while the true ك remains a strong posterior alternative. On the baseline, this produces a false gap at 52:26:3; the desired location is 52:26:2. The exact stronger posterior fixture failed on baseline with that location mismatch in Modal run `gap-ownership-red-green-4aaa7ab-dirty-20261031T114312Z`, shard 0.

An otherwise identical weaker posterior fixture gives an 8.53-nat advantage, below the unchanged 10-nat margin. Its existing placement must remain unchanged. This is an explicit abstention control, not a claimed fix. Additional controls cover clean recitation, harmless onset noise with the short word actually spoken, two adjacent omissions, a genuine wrong word beside an omission, and the existing noisy-gap, addition, repeat, held-sound, and self-correction suites.

Final Modal verification passed in `gap-ownership-final-4aaa7ab-dirty-20261031T114410Z`: the core TypeScript build, all 45 existing checker tests (split over three shards), 6 noisy-gap tests, 8 addition tests, and 7 new ownership tests. Both the strong correction and weak abstention assertions passed. The only test-harness change during verification was an explicit 30-second timeout for the two posterior-heavy synthetic tests; their actual runtimes were about 4.6 seconds each.

## Limits to assess

- Requiring clean outer visits deliberately misses harder cases with several neighboring errors.
- A confidently wrong posterior can still prefer the wrong ownership; clean-session false warnings therefore remain a primary outcome.
- The posterior comparison adds work only when a guarded gap candidate exists. Full practice runtime and old/new result concentration need measurement.
- Fixture development established the mechanism, not generalization. The preregistered practice comparison decides whether it helps.

The parent run will record the final practice counts and verdict. This branch is neither folded nor discarded.
