# Faithful splice training

Status: complete; reject. No fold or discard performed.

## Prediction and fixed plan

Predict five fewer listener-attributed wrong calls and four fewer overall wrong calls, primarily by recovering at least
three added-word misses. Kill the experiment if it reproduces the recurring phone false warnings from the earlier
channel-adapted arms or recovers fewer than three added-word mistakes.

Start from the shipped `spliced-clean` checkpoint and fine-tune for one seed-0 pass over at most 50 hours of the existing
gold-excluded `clean-match` training pool, capped at 200 updates. Keep the architecture, greedy decoder, spliced target
for untouched clips, KD weight 2, learning rate 0.0015, 800-second batch budget, and the shipped noise/RIR/gain/SpecAugment
chain. Replace half of sampled training clips with deterministic unfaithful edits, weighted 2:1:1 across insertion,
word-like substitution and deletion. Use the final scheduled checkpoint, not practice results, to select the model.

## Mechanism

Each edit uses real audio from another utterance by the same training-speaker cluster. Stored teacher CTC peaks define
contiguous phone boundaries. Insertions go immediately before the host's first emitted phone, after its leading pause;
substitutions and deletions leave at least three emitted phones at both clip edges. The CTC target is the actual sequence
assembled from the host and donor greedy transcripts, not the canonical Quran reference.

The old teacher posterior is retained only on unchanged host audio. Output frames are mapped back to their original host
sample positions after a length-changing edit; KD is masked across the edited audio and an 80 ms coarticulation margin.
This isolates the proposed lesson—emit acoustically present out-of-sequence speech—without adding a phone-channel arm.

## Validation

- Pure splice tests cover CTC run alignment, insertion/substitution/deletion targets, length changes, and post-edit KD
  remapping/masking. All 20 student/splice tests pass locally and in a one-shard Modal run.
- A two-update H100 smoke run initialized from `spliced-clean` step 30,287, loaded the existing augmentation banks,
  trained with finite CTC+KD loss, saved/resumed the adaptation metadata, and completed unchanged clean evaluation.
- The fixed candidate completed 200 updates over 21,161 clips / 50.0008 hours. Logged batches contain all three edit
  classes and finite CTC+KD losses. Its 300-clip clean diagnostic raw PER is 10.30% versus the frozen teacher's 8.05%;
  this regression diagnostic was not used for selection.
- Trace of `lab-v1.5-p07` at 144.12 s shows the planted added word `سَلِيمٍ` at an ayah-start location. The shipped
  listener emits `س َ ِ ۦ` but loses the final `م`; checker-threshold and local-window variants do not recover an
  added-word warning. This is consistent with incomplete listener evidence, not a checker threshold miss.

Limits: phone-span chunks approximate words because the immutable training manifest retains phoneme transcripts but not
word locations. Same-speaker selection avoids voice-switch shortcuts, but synthetic seams remain. Teacher greedy labels
can contain its original acoustic mistakes; the experiment only trusts them as as-heard labels at sites where the stored
top-1 posterior exactly reproduces the stored transcript. Clean PER is diagnostic only and does not select a checkpoint.
The fp32 ONNX export has two collapsed-phone edits across six parity clips (PER 0.37% versus chunk-masked PyTorch), both
on one 593-frame clip; the final decision therefore uses the shipped stateful int8 path, not PyTorch.

The candidate int8 model is 67,759,744 bytes, SHA256
`8bc8d34ffeb41a35023649349cd85c8e90687ffda8f0e22649647df22c8fea2c`. All 400 available lab-v1.5 phone/studio
session files decoded successfully under the separate `faithful-splice-int8` listener id.

## Result

The one preregistered practice score decisively fails:

| Practice phone | Main | Candidate | Change |
|---|---:|---:|---:|
| Wrong calls | 89 | 229 | +140 |
| Missed mistakes | 64 | 202 | +138 |
| False warnings | 25 | 27 | +2 |
| Listener-attributed wrong calls | 47 | 182 | +135 |
| Listener PER | 4.88% | 7.00% | +2.12 points |

The candidate recovered six phone mistakes but newly missed 144. All eight preregistered added-word incidents remain
listener-attributed misses, so it recovers zero of the required three. The warning identities also churn: new phone false
warnings appear, independently failing the clean-warning gate. Studio is much worse as well: 45 misses + 13 warnings
becomes 160 + 24.

This rejects the recipe, not merely its expected effect. Phone-span edits cut arbitrary within-word sequences rather than
semantically parallel whole words, and half of continuation batches is too destructive for this checkpoint: the model
loses broad long-session evidence despite keeping finite losses and only a smaller short-clean diagnostic regression.
Do not tune the splice rate or update count against this practice result. A future attempt needs word locations retained
in a new training-data version, paired same-seam controls, and a held-out synthetic/real-error development gate before
another practice score.
