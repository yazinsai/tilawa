# Natural untranscribed speech relabel

Status: fine-tuning and validation in progress.

## Prediction

Listener −3 wrong calls and overall −3 on practice, with −2 as the honest floor. The card predicts recovering at least three of eleven residual added-word targets (seven incident keys are explicitly listed). Kill if fewer than three targets recover after fine-tuning, or if clean phone/studio pauses acquire any new word-shaped decode.

## Mechanism and source audit

The shipped `spliced-clean` checkpoint trained for twelve epochs on 225,401 gold-excluded `clean-match` utterances (534.82 hours), with canonical phones inside teacher anchors, teacher phones at clip edges, and free teacher top-five KD at weight 2. The original claim that both losses force all untranscribed speech to blank is too broad: KD is already unconstrained, and edge speech is already preserved. The remaining testable conflict is canonical CTC suppressing interior extra speech that the teacher actually emitted.

A complete census of that exact training population found 2,113 clips with anchored, word-shaped teacher insertions. Of 2,106 insertions with consistent cached transcripts, 1,782 had two confident consonants and 1,133 had two consonants with posterior at least 0.9 on frames the canonical CTC alignment leaves blank. The census excluded the same 150 gold clips as the shipped training run. Thirty-two candidate clips had cached transcript/posterior disagreements and were skipped rather than relabelled.

Mining uses the saved free teacher pass, not a reference-conditioned teacher decode. It requires an interior insertion between exact phone anchors, at least three phones and two consonants, and Silero VAD speech at canonical-blank consonant frames. Every phone must reach 0.9 confidence for relabelling. Other qualifying speech keeps the original CTC target but masks its canonical-blank speech frames from both CTC and KD. VAD-negative pauses are untouched. A constant blank bridge in the CTC lattice prevents gradients on masked frames while preserving the time axis and repeated-label separation.

VAD mining retained 776 confident spans in 775 clips and 589 uncertain spans in 585 clips. The source parquet files are read-only. A separate overlay records corrected targets and frame masks, with base-target and frame-count checks when loaded. The planned fine-tune resumes the shipped model, optimizer and scheduler for exactly two more epochs over the original full dataset, retaining architecture, noise/RIR/gain/SpecAugment and KD weight. Export uses the existing streaming graph and dynamic int8 MatMul recipe.

## Validation so far

- Complete training census: supervisor Modal run 4, six shards.
- Core and SDK build/tests: supervisor Modal run 3, all six shards passed.
- Nine synthetic label, pause, cache-disagreement, masking-gradient and overlay tests passed (supervisor Modal run 7).
- VAD mining: supervisor Modal run 5, six shards.
- Fixed two-epoch H100 fine-tune: supervisor Modal run 8.
- Baseline-against-itself crop diagnostic: supervisor Modal run 9 (pending).
- No practice score has been requested.

## Limits

Teacher confidence and VAD are mining filters, not independent verification that the pseudo-label is correct. No calibrated independent listening service was used. The census covers anchored teacher insertions; speech suppressed by the teacher itself is outside its reach. Post-training crop diagnostics use labelled intervals to test recovery and pause safety; they do not establish end-to-end warning gains. Only the unchanged lab scorer can establish those gains.
