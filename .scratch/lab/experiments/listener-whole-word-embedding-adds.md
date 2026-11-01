# Whole-word acoustic embeddings for uncertain additions

Status: experiment in progress; no practice score taken.

## Prediction and kill criterion

Preregistered listener −3 wrong calls and overall −3 on practice, with an honest
floor of −2. The proposed wins are the uncertain additions at p10 26:164:1,
p07 37:36:1 and p33 26:208:1. The other two cluster incidents are speculative.
Kill if fewer than two of the three primary targets confirm in checker replay,
or if a suspected addition on any fully clean session confirms.

## Mechanism

Freeze the shipped `spliced-clean` student's first three encoder stacks and
learn a small temporal convolution, attention pooling and projection head. Its
128-dimensional unit vectors should identify a complete word even when the
phoneme CTC head loses a weak onset. The objective is supervised contrastive
learning: positives share the mushaf word's isolated pronunciation and come
from different original recordings. No synthetic audio splices, insertion
examples, new channel augmentations or practice examples enter training.

Canonical transcripts locate words in the complete reference. Forced alignment
uses the existing teacher posteriors on gold-excluded clean-match s1 clips.
Real word crops retain up to 80 ms at each edge, bounded by the halfway point to
the adjacent phone. The shipped noise, reverb, gain and SpecAugment chain
provides two augmented views alongside each unaugmented crop. Training and
calibration use the existing train and val splits respectively.

Training-word prototypes form the whole-word bank. The acceptance radius is
the 99th percentile of held-out real words' cosine distances to their matching
prototype, using the higher empirical quantile. It is fixed before practice
queries are evaluated. Neighbour alternatives include all seven words within
±3 of the checker anchor and their proper prefixes. Prefix exemplars pool the
corresponding leading features of real training words; they do not enter the
training loss. Missing neighbour coverage causes abstention.

Only an already-suspected added word with sufficient decoder confidence and
failed CTC support can request this evidence. The proposed listener crop covers
the whole anchored gap, including speech omitted by greedy decoding. Ordinary
energy endpointing removes edge silence with the same 80 ms margin as training.
A non-neighbour whole word must win outright over the neighbour/prefix bank
and lie inside the fixed radius. The decoded phoneme string is unchanged.

## Validation so far

Supervisor Modal run 2 replayed the three primary targets on the folded
`exp-listener-context-free-gap-decode-62ac8b5` evidence. All three remain
uncertain with shipped policy and confirm with the CTC local-window gate off.
Their mean decoder confidences are 0.883, 0.665 and 0.804 respectively. The
observed strings are `كَرِۦۦم`, `فَل` and `فَرِۦۦم`. This supports testing a
whole-gap acoustic representation: the second target's decoded span contains
only an onset, not the whole planted word.

The initial pipeline run (Modal 4) completed 60,000 training segments, 5,000
validation segments and 5,000 optimizer steps. Its bank contained 7,653 words;
13,353 held-out views had 76.60% top-1 retrieval and a 99th-percentile distance
of 0.855012. This model was not evaluated on practice. Before practice
evaluation, construction fixes admitted short words needed as neighbour
alternatives, included one-phone prefixes and made temporal pooling exportable.
A fresh run with these fixes retains the same data limits and training steps.

The first all-session diagnostic job (Modal 5) timed out. It is not evidence of
complete clean-session coverage or a kill-criterion result. The replacement
uses 20 shards and obtains gap bounds from the checker's actual support
alignment. The diagnostic reads saved evidence and uses actual checker
decision callbacks; it does not modify evidence, labels or the scorer.

The corrected calibration, complete replay and test results will be recorded
here when the jobs finish.

## Limits

Prefix exemplars use leading frozen features from a complete real word rather
than a separate encoding of every possible restart. The finite real-word bank
cannot provide exemplars of every Quran pronunciation; uncovered neighbours
must abstain. True-position checker replay tests the proposed evidence decision,
not the full free-recitation release gate. No overall score can be inferred
from a query count.
