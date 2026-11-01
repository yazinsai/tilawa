# Whole-word acoustic embeddings for uncertain additions

Status: implemented; the exported SDK model passes the kill replay (3/3 primary targets, zero clean confirmations). No official practice score taken; supervisor scoring and the fold gate remain pending.

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
±3 of the checker anchor and their proper prefixes. Each prefix of a real
training segment is encoded independently so the frozen encoder cannot look
ahead into its unseen suffix. Prefixes do not enter the training loss. Missing
neighbour coverage causes abstention.

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
The reference prefix bank also uses independent encodes of real prefixes,
instead of truncating features that already saw the word's suffix. This bank
can be rebuilt without changing the learned head or calibration radius.

The reference has 15,905 distinct nonempty isolated word forms. A separate
gallery pass searches all gold-excluded clean-match training shards for up to
four real exemplars of each form absent from the bounded training sample.
These extend the acoustic reference bank without changing the learned head,
the existing prototypes or the held-out radius. Missing forms still abstain
when they are neighbours. The reported held-out top-1 statistic describes the
original training bank; its bank size is retained explicitly in the report.

The first all-session diagnostic job (Modal 5) timed out. It is not evidence of
complete clean-session coverage or a kill-criterion result. The replacement
uses 20 shards and obtains gap bounds from the checker's actual support
alignment. The diagnostic reads saved evidence and uses actual checker
decision callbacks; it does not modify evidence, labels or the scorer.

Modal 7 passed the core and SDK builds/tests on all 20 shards and collected
exactly 200 unique session/variant pairs (100 sessions, phone and studio).
There were 13 CTC-rejected addition queries, eight with an anchored gap. None
of the queries came from the 26 fully clean sessions (52 variant cases). All
three primary targets had a gap. These are query eligibility counts, not
embedding confirmations or a practice score.

The corrected training run (Modal 8) used 60,000 train and 5,000 validation
segments from 14,320 and 1,237 recordings respectively, with zero shared source
clusters. It trained for 5,000 steps. The original 7,651-word bank gave 76.35%
held-out top-1 retrieval. The 13,353 calibration views set the radius to
**0.802246630191803**, before practice replay. Modal 10 extended the gallery
to **15,833 of 15,905** forms using 31,748 additional real training segments,
with **393,825** independently encoded prefix references. The head and radius
were unchanged. The 76.35% retrieval figure predates this gallery extension.

Modal 10 replayed all 200 session/variant cases using the final gallery. Eight
of the 13 uncertain-addition queries had an anchored gap. Five had evidence
actually changed by the folded cropped decoder; all five confirmed. The
other three abstained under the provenance guard. None of the 52 clean cases
had an eligible query, so clean confirmations were zero. This is evidence
about the query gate; it does not estimate the embedding's false-positive
rate on arbitrary clean pauses.

| Primary target | Whole-word distance | Best neighbour/prefix distance | Confirmed |
| --- | ---: | ---: | --- |
| p10 phone, 26:164:1 | 0.02688 | 0.76965 | yes |
| p07 phone, 37:36:1 | 0.21688 | 0.66880 | yes |
| p33 phone, 26:208:1 | 0.13077 | 0.58286 | yes |

The additional confirmations were p08 phone and p79 studio. They are not
claimed as additional score wins without an end-to-end score. The p79 phone
gap was unchanged by the folded path and abstained; p19 had no query. Thus
the preregistered kill replay gives **3/3 primary confirmations, zero clean
confirmations**. The nearest whole word was correct at p10, but p07's nearest
prototype was a different word, and p33 preferred an article-bearing variant.
The result supports an added-word decision more strongly than exact lexical
recognition.

The SDK integration packages the frozen encoder and head as a separate ONNX
graph alongside the unchanged CTC graph. It replays the folded evidence on
the canonical frame grid with the free tracker, asks only about CTC-rejected
additions in changed gaps, and stores successful word comparisons on the
corresponding decoded phone. The checker requires the exact anchor and phone
span, a non-neighbour lexicon word, and both calibrated-distance and prefix
veto conditions before using the proof. No scorer changes are needed.

As with the existing folded crop pass, this enrichment runs at session end
when evidence recording is enabled; it changes subsequent evidence replay,
not the first-pass live events. Extending that existing path to online
asynchronous confirmation is outside this experiment.

Modal 17 passed all core and SDK builds and tests: 640 core tests and 21 SDK
tests, with one existing SDK integration test skipped. The new tests cover
span and anchor matching, radius and neighbour/prefix vetoes, missing gallery
coverage, silence, and unchanged decoded phones/posteriors. The Python
numerical and data-construction tests also passed during the training runs.

Modal 23 exported the frozen middle encoder and head to ONNX. The exporter
retains cold-start attention masks, dynamic edge padding, and the fixed chunk
size of each convolution stack; icefall's normal tracing path assumes a
stateful streaming export and otherwise drops these semantics. Six crop
lengths (14, 31, 50, 123, 203 and 397 fbank frames) match the original eager
PyTorch model, with maximum absolute embedding error **3.912e-7**. The exporter
also checks that its frozen checkpoint matches the one used for calibration.
The ONNX SHA-256 is
`2cb8353827b4929c84e335775da79d3b9aafe53ff21ace48fd7550032e97ad05`.
The packaged CTC graph is byte-identical to the shipped graph.

The bundle is at `/scratch/listener-whole-word-embedding-adds/model`, selected
by `.autolab/model.json`. It contains the unchanged CTC model, a separate
77,098,954-byte word encoder, and hashed real-word/prefix galleries. Node's
directory loader loads the sidecar automatically. Other SDK backends can use
the exported comparison interface but do not yet fetch sidecars automatically.
Total bundle size is **356,563,951 bytes (340.05 MiB)**, above the 200 MiB device
release budget; this experiment does not claim release readiness.

Modal 24 exercised the packaged model through Node's actual loader and
`confirmGapWords`, using the free tracker to generate queries from the saved
folded listener evidence. It then replayed the annotated evidence with both
the free tracker and known positions. All **200 session/variant cases**
completed (100 phone and 100 studio), and all 20 shards exited successfully.
There were **five whole-word proofs and zero proofs on the 52 fully clean
cases**. The five proofs were at the same sites as the Python replay (the
three primary phone targets, p08 phone and p79 studio); all five matched
planted additions and confirmed in both checker replays. Their primary-target
CPU whole-word distances were 0.026888, 0.216401 and 0.130912, respectively;
the fixed radius remained 0.802246630191803. Modal 28 independently checked
the completed output counts and shard statuses while the full report was
being downloaded.

The preregistered kill criterion therefore did not fire. These are acoustic
and checker replay results, not a measured change in overall wrong calls.
The implementation remains in place for the supervisor's own decode, score,
reader holdout check and fold decision.

## Limits

The finite real-word bank cannot provide exemplars of every Quran pronunciation;
uncovered neighbours must abstain. Prefixes use up to four real exemplars per
word and cannot exhaust every possible restart. Saved-evidence replay tests
the acoustic comparison and checker integration. It does not replace the
supervisor's fresh audio decode and full free-recitation fold gate. No overall
score can be inferred from a query count. No official practice score was
requested, and no calibration or acceptance constant was fitted on practice.
