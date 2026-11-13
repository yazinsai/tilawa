# Moving spectral zero pronunciation evidence

Status: needs-human — implementation and local checks complete; authorized
supervisor execution failed before training, target inspection or remote tests.
There is no fitted real-data bundle. This draft is not ready for practice scoring.

## Prediction and premise

Predict listener −1 and overall −1: one false wrong-word warning removed, with
no predicted losses. A stationary recording notch can resemble an articulatory
antiresonance in individual frames. The proposed discriminator is its trajectory:
a recording zero should have a common location throughout a window, while an
articulatory zero can move with the pronunciation.

The card requires moving evidence to prefer the complete written pronunciation
while a matched static pole-zero control still prefers the already-accused rival.
The task additionally requires three target warnings removed before population
scoring. No target result is available. Infrastructure failure does not establish
that the premise is wrong, that there are zero wins, or that the kill criterion
fires. The nasal-antiresonance branch was absent from this checkout; this draft
implements its own rational extractor and uses the existing SDK evidence route.

## Acoustic model

`packages/sdk/src/moving-spectral-zero.ts` observes the primary frontend's
25 ms Povey-windowed, DC-removed, preemphasized spectrum. It selects the first
native analysis in each 40 ms evidence cell, anchored at sample zero. Spectrum
observations are copies: a side scorer cannot mutate the primary fbank.

A relative power floor and 16 retained cepstral coefficients produce a smooth
minimum-phase transfer function. Three fixed weighted linear solves fit
`A(z) H(z) = B(z)`, with denominator degree eight and numerator degree six and
a fixed 0.01 ridge. Simultaneous polynomial roots provide four pole pairs and
three zero pairs. Minimum-phase representatives use normalized angle and
log-radius bandwidth. Real roots are paired on the real axis; unresolved roots
mark a frame's zero observations unavailable. Eight symmetric pole summaries
include frequency moments, bandwidth moments, their product and log energy.

Each zero observation has six possible assignments: one recording zero and
two ordered latent articulation zeros. The forward sum integrates every
assignment, rather than selecting a best root track. Three window-derived
center hypotheses approximate integration over the stationary zero's location;
one center choice is shared by every frame in the window. Both pronunciations
and both arms use identical center hypotheses and stationary variances.
Centers depend on acoustic observations, never candidate text.

The moving arm uses a phone-conditioned diagonal AR(1) density for the two
articulation zeros, conditioned on the preceding frame's latent correspondence.
Its sixteen zero parameters per phone are four means, four initial variances,
four autoregression coefficients and four innovation variances. The static arm
uses an equal-weight two-component diagonal mixture over instantaneous zero
positions: eight means and eight variances, also sixteen parameters per phone.
It sums assignments independently at each frame and has no trajectory term.
The pole emission parameters are shared: they are fitted with the static arm,
then frozen during dynamic fitting. Neither arm gets extra pole capacity.

The joint forward sum also marginalizes all phone boundaries using ordered
CTC-style phone and optional blank states, a shared 0.8 geometric stay prior,
and ordinary repeated-phone separation. Only a complete pronunciation accepts.
Subtracting the identity-neutral timing partition conditions on observed length.
Missing zero observations supply no zero density and reset the next transition;
pole observations remain available. Unknown/unseen phones, impossible complete
placements, malformed observations, all-unresolved zero windows and windows over
2048 observations abstain. Ties and nonfinite scores cannot certify a veto.

## Training protocol

`modal/student_moving_zero_extract.ts` requires an explicit speaker exclusion
file, unions all non-training speakers found in label metadata, and accepts
only help-train none-labelled training takes. It rejects contradictory mistake
or repeat labels, non-single-ayah takes and paths escaping the supplied root.
The canonical final word uses the supplied isolated pronunciation. This assumes
the label fields `speaker`, `split`, `kind`, `wav`, `ayahs`, `events` and
`non_errors`; the unavailable data audit has not confirmed that actual layout.
Release-held-out exclusion provenance also remains unverified in the container.

`modal/student_moving_zero.py` performs eight fixed soft-EM passes per arm.
Posterior state and correspondence occupancies estimate Gaussian statistics;
dynamic updates also integrate successive correspondence pairs. Estimates shrink
toward pooled training moments. The autoregression prior is 0.8, its magnitude
is bounded below one, and variance floors prevent degenerate fits.

The only selection grid is shrinkage 16 or 64. Three hash-ordered, speaker-disjoint
training folds compare both arms on identical deterministic single-phone rivals
derived from training pronunciations. The objective is the speaker-balanced
average contrast log loss of both arms; ties favor stronger shrinkage. The final
configuration is refitted once on eligible training recordings and exported with
an input hash, exclusions, fold membership, validation losses and EM history.
These choices were specified before accessing targets. **The grid is frozen;
no configuration has yet been selected or fitted on real recordings.**

Extraction reserves 30 CPU minutes and fitting/validation reserve the remaining
23.5 CPU hours, checked between recordings. Real throughput and peak memory
have not been measured. The training estimator is CPU-only; no student weights,
primary vocabulary, model manifest or training augmentation are modified.

## Engine integration

The checker exposes both complete pronunciations and the exact native-frame
window used by an otherwise confirmed string-path wrong-word decision. Graph
decisions are exempt. The SDK compares both arms on identical observations
and supplies a certificate only for a strict dynamic-written/static-rival
reversal. Every certificate binds the word, both complete strings and both
frame endpoints. A mismatched frame count or changed pronunciation abstains.

A veto changes only the public wrong-word confirmation to uncertain and removes
that ID from the public mistake list. The internal issue/confirmation ledger
retains baseline state, preserving later insertion merges and ownership decisions.
Primary decoding and tracker input remain unchanged. Live audio sessions evaluate
the side scorer before publishing confirmation. Recorded evidence carries
certificates on the first primary token after existing gap and teacher enrichment.

The optional Node loader verifies the side asset's fixed filename, length,
SHA-256, inventory, clock and parameter shapes. Declared missing/corrupt assets
fail loading and release the backend. The inherited insertion teacher still
loads through its original route. No new registry or primary manifest was edited.

`modal/student_moving_zero_probe.ts` accepts arbitrary SDK audio/evidence paths
and a frozen side model. It discovers actual engine comparisons using the free
tracker, records every comparison, and replays baseline and enriched evidence.
It asserts unchanged tracker events and permits only confirmed-to-uncertain
wrong-word changes. It does not import the lab scorer, oracle positions or labels.
No registered session, reader or verse selector appears in runtime/training code.

## Validation and execution block

Three requests used only the authorized supervisor route:

1. Read-only help-train layout, label and available-artifact audit.
2. Minimal retry: container working directory, help-train listing and label header.
3. Supplied six-shard build/core/SDK harness plus Python numerical tests.

Every request returned `NotFoundError: No such file or directory` before any
container output. No training data or target audio was accessed. The user was
asked to restore execution while implementation continued; no alternative Modal
route or release exam access was attempted.

Local checks passed:

- Core and SDK TypeScript builds; strict checks of the extractor, probe and new tests.
- Six core certificate tests: complete comparison discovery, strict reversal,
  JSON transport, exact endpoint/string binding, retained issue creation,
  neighboring addition preservation and graph confirmation preservation.
- Eleven new SDK tests and three existing teacher-loader tests: root extraction,
  chunk invariance, unchanged primary fbank, marginalized correspondence,
  temporal-order discrimination, completion/abstention, matched control,
  verified/missing/corrupt assets, live-session veto and primary evidence retention.
- Full local SDK suite: 36 tests passed, one data-dependent help test skipped.
  This includes real ONNX execution of the tiny listener fixture, existing audio
  sessions, evidence chunking, Node tier loading and insertion-teacher behavior.
- Four Python numerical/provenance tests. Joint forward sums agree with an
  independent exhaustive boundary/root/nuisance path enumeration; frame and
  correspondence-pair posteriors normalize. Tiny soft-EM fits are finite and
  both arms have equal parameter counts. Excluded speakers and mistake/repeat
  labels are rejected. Relative flooring preserves rational roots under gain.
- Python/SDK parity on two synthetic frames and twelve complete-pronunciation
  scores, including a tiny fitted model accepted by SDK validation. Worst root
  coordinate/pole-feature difference was `1.36e-11`; worst score difference was
  `1.16e-14`. These fixtures are ignored local check files, not trained assets.
- Python syntax compilation and `git diff --check`.

The full remote suites remain unverified. Toy examples establish implementation
properties, not acoustic validity on Quran recordings or population safety.
No practice score was requested; all three allowances remain. The card, scorer,
lab tooling, data, evidence and supplied manifests remain untouched. No fold
or discard occurred. Target wins, listener delta and overall delta are unknown.

## Resume and limits

Restore supervisor execution and first confirm label layout, exclusions and the
inherited listener bundle. Run the extractor and trainer in the authorized CPU
container, persisting outputs under `/scratch/listener-moving-spectral-zero-evidence`.
Inspect the frozen training report before accessing target comparisons. Use the
generic probe with the inherited listener evidence and actual phone WAVs to
measure whether each registered warning is removed on the ordinary free path.

If fewer than three targets are won, revert the draft engine changes, retain the
measured findings, commit the premise kill and write the specified killed outcome.
If at least three are won, assemble and verify a complete bundle containing the
unchanged primary ONNX/manifest, inherited teacher assets and both side asset
files. Only then declare untracked `.autolab/model.json` and request practice
scoring before considering narrowing from gate results.

There is presently **no real-data side asset, no complete experimental bundle
and no `.autolab/model.json`**. Running the inherited main model cannot evaluate
this change. The untracked outcome records the infrastructure block without
inventing a kill or a zero-win measurement.

Minimum-phase root estimates are not necessarily physical vocal-tract zeros.
Spectral liftering, real-root pairing, the degree cap and local rational fits can
move or merge roots. Three acoustic center hypotheses provide coarse nuisance
integration; stationary articulation can also resemble a recording notch.
Diagonal AR dynamics and sparse none-labelled phone coverage can favor a written
sequence on a real substitution. Positive written preference and a static reversal
do not prove preservation of mistakes. Live extraction duplicates spectral work,
retains observations/audio and replays certificate discovery; device cost is
unmeasured. Node installation is implemented; web/native bundle installation is not.
