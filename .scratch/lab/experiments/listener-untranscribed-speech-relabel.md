# Natural untranscribed speech relabel

Status: fine-tuning, export and pause validation complete; practice scoring in progress.

## Prediction

Listener −3 wrong calls and overall −3 on practice, with −2 as the honest floor. The card predicts recovering at least three of eleven residual added-word targets (seven incident keys are explicitly listed). Kill if fewer than three targets recover after fine-tuning, or if clean phone/studio pauses acquire any new word-shaped decode.

## Mechanism and source audit

The shipped `spliced-clean` checkpoint trained for twelve epochs on 225,401 gold-excluded `clean-match` utterances (534.82 hours), with canonical phones inside teacher anchors, teacher phones at clip edges, and free teacher top-five KD at weight 2. The original claim that both losses force all untranscribed speech to blank is too broad: KD is already unconstrained, and edge speech is already preserved. The remaining testable conflict is canonical CTC suppressing interior extra speech that the teacher actually emitted.

A complete census of that exact training population found 2,113 clips with anchored, word-shaped teacher insertions. Of 2,106 insertions with consistent cached transcripts, 1,782 had two confident consonants and 1,133 had two consonants with posterior at least 0.9 on frames the canonical CTC alignment leaves blank. The census excluded the same 150 gold clips as the shipped training run. Thirty-two candidate clips had cached transcript/posterior disagreements and were skipped rather than relabelled.

Mining uses the saved free teacher pass, not a reference-conditioned teacher decode. It requires an interior insertion between exact phone anchors, at least three phones and two consonants, and Silero VAD speech at canonical-blank consonant frames. Every phone must reach 0.9 confidence for relabelling. Other qualifying speech keeps the original CTC target but masks its canonical-blank speech frames from both CTC and KD. VAD-negative pauses are untouched. A constant blank bridge in the CTC lattice prevents gradients on masked frames while preserving the time axis and repeated-label separation.

VAD mining retained 776 confident spans in 775 clips (778.56 s) and 589 uncertain spans in 585 clips (357.76 s masked). The overlay covers 1,352 distinct clips. The source parquet files are read-only. A separate overlay records corrected targets and frame masks, with base-target and frame-count checks when loaded. The fine-tune resumed the shipped model, optimizer and scheduler for exactly two more epochs over the original full dataset, retaining architecture, noise/RIR/gain/SpecAugment and KD weight. It completed 5,046 updates, from step 30,287 to 35,333, with no non-finite skips. Export uses the existing streaming graph and dynamic int8 MatMul recipe.

## Validation so far

- Complete training census: supervisor Modal run 4, six shards.
- Core and SDK build/tests: supervisor Modal run 3, all six shards passed, 656 tests total.
- Twenty-one Python student tests passed, including ten new label, pause, cache-disagreement, masking-gradient, overlay and CTC-only compatibility checks (supervisor Modal run 17).
- VAD mining: supervisor Modal run 5, six shards.
- Fixed two-epoch H100 fine-tune: supervisor Modal run 8, completed in 44 minutes. Raw reference PER on the same 3,000 held-out training-evaluation clips changed from 9.38% to 8.73%. This is a training-evaluation result; no unchanged-label continuation control was run.
- Baseline-against-itself crop diagnostic: supervisor Modal run 11, 166 crops identical across the two passes; all 158 clean-pause crops remained below the word-shape bar.
- Streaming export and dynamic int8 MatMul quantization: supervisor Modal run 18. The 67,759,744-byte model passed interface equality and SHA-256 checks. The existing fp32 PyTorch/ONNX check had one phone edit across six clips (0.259% PER); five clips had identical greedy phones.
- Full crop comparison: supervisor Modal run 19, all 20 shards passed. No newly word-shaped pauses among 1,054 phone and 1,054 studio clean-pause crops (four existing shapes in each variant remained four). Of 118 added-word crops, baseline/candidate shapes were 111/102, with two gains and eleven losses. Six of seven supplied targets already had a word shape in exact label crops under both models; this oracle crop does not measure whether the engine finds that gap or confirms a warning.
- First full practice score requested after pause validation.

## Limits

Teacher confidence and VAD are mining filters, not independent verification that the pseudo-label is correct. No calibrated independent listening service was used. The census covers anchored teacher insertions; speech suppressed by the teacher itself is outside its reach. Post-training crop diagnostics use labelled intervals to test recovery and pause safety; they do not establish end-to-end warning gains. Only the unchanged lab scorer can establish those gains.
