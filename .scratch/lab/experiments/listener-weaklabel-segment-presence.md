# Whole-recording weak-label segment presence

Status: implementation and training validation in progress. The experiment card
is unchanged; no fold has been performed.

## Prediction and mechanism

The card predicts two omission false warnings removed and one caught omission
lost: listener −1 and overall −1. The three registered targets are p50
37:113:10, p65 4:12:40 and p65 4:15:6. The card's premise bar is two complete-read
preferences; the execution brief requires three target wins to retain the built
change and request a practice score. That stricter stopping rule applies.

The small acoustic/text segment CRF reads frozen intermediate encoder features
from the shipped int8 student. An existing third-stack tensor is exposed without
changing its weights or primary outputs. When its rate exceeds the native
40 ms output clock, adjacent feature frames are mean-pooled to that clock.

Each word is described by its phoneme histogram, first and final phones, hashed
phone bigrams and log phone count. Separate 16-dimensional acoustic and text
projections produce frame/word compatibility. A learned geometric segment
duration scales with phone count and recording rate. Every read word consumes
at least one frame; the specified omitted word consumes exactly zero frames.
All monotone boundaries are summed, with no forced word cuts or maximum segment
duration. A prefix log-sum recurrence computes that sum in O(words × frames).

Training uses entire usable single-ayah none-labelled and single scripted
word-omission recordings from help-train's training speakers. Repeats, other
mistakes, unusable takes and evaluation speakers are excluded. There are no
excision negatives, engine visits, forced alignments or practice examples in
training. The objective contrasts the known label path with the normalized union
of complete reading and every possible single-word omission. A learned omission
prior is uniform over written words. It cannot skip two words in one path.

Three folds group recordings by speaker. Feature normalization and fitting are
performed inside each training fold. Mean held-out label NLL selects L2 from
0.001, 0.01 and 0.1. Architecture, seed, 20-epoch schedule, optimizer and candidate
decision rule are fixed before target evaluation. A final model is fit on all
eligible training recordings and frozen before practice diagnostics.

At an already-proposed single-word omission, read-only discovery captures the
checker's existing ayah interval. Complete read and exact omit explanations use
the same frozen features, word scores, duration distribution and interval.
Strict preference for reading supplies an exact-span listener certificate and
vetoes confirmation. Ties and unavailable evidence preserve the existing
decision. No primary tokens, tracker events, placement or confirmation constants
are changed.

## Validation and limits

Python and SDK tests enumerate short sequences' boundaries to check the forward
sum, mandatory nonempty read segments, zero-frame omissions, and consumption of
every acoustic frame. Python additionally checks the union contains exactly one
copy of each complete or single-omit explanation and has finite training
gradients. Input eligibility checks reject evaluation, excluded and repeat takes.
Core tests cover unchanged discovery, exact-span certificate transport through
JSON, strict read preference, stale or invalid certificate rejection, and
preservation of suspicion, location and timestamp when confirmation is vetoed.

Modal run 2 extracted 384-dimensional frozen features and trained on 206
recordings from 78 speakers: 150 none-labelled and 56 scripted single-word
omissions. The final scorer has 8,723 trainable parameters. Speaker-grouped
held-out mean label NLL was 0.495960 for L2 = 0.001, 0.495588 for L2 = 0.01 and
0.495924 for L2 = 0.1. Thus the training-only selection chose 0.01, with little
difference between regularization choices. The final learned duration parameters
were [0.195139, −0.202339] and the omission log prior was −0.502144. No parameter
was changed after inspecting these results.

Frozen feature graph SHA-256:
`ae3d060f3211c9c893dc8acb638bc8fb6009ad39f5e4b1ef06d37932c1d8992e`.
Final scorer JSON SHA-256:
`03dadd5123627541b5328eeccd623ef2af04b5108415f1614699a2c2d959d8aa`.

Run 3 built both packages and passed all existing core checks. One of the four
new core tests timed out at five seconds while replaying six fixtures, without
an assertion failure. Its timeout was increased to 20 seconds; run 4 passed
all four new core tests, all 29 SDK tests (one optional integration test skipped)
and all five Python tests. The existing core RPC timeout after passed tests was
accepted only through the supplied harness's documented exception.

The first target preparation failed before feature extraction because the
supervisor source snapshot excludes the experiment card. The diagnostic was
relaunched with its unchanged listener and set values passed as command
arguments. This changed no training, engine behavior or supplied files.

Target results are pending. No practice score has been requested.
The supervisor's fresh SDK path and complete declared model bundle must be
verified if the target premise survives; a primary-only run cannot evaluate
this experiment.

The weak labels remove dependence on word-cut supervision, but do not guarantee
correct latent segmentation. Neighboring speech can still explain a missing
word. Single-ayah learner recordings differ from long free sessions, geometric
durations include silence assigned to word segments, and the written forms may
miss pronunciation variants. These are limitations to measure, not reasons to
stop because of collateral losses. The available teacher is correlated evidence,
and no independent listening judgment or test-label correction is claimed.
