# Sparse consonant mining with matched random reweighting

Status: diagnostics and verification complete; v1 superseded before practice scores; exposure-matched v2 preparing. No fold, discard, or exam scoring.

## Fixed experiment

Original shipped initialization and ordered seed-0 50-hour clean-match training pool; original single-view CTC1 + KD2; 75% phone augmentation; 200 updates; learning rate .0015; 800-second batch budget; same scheduler and final scheduled checkpoint. No consistency term or extra clean-view loss. Both arms use the original frame-normalized CTC/KD weighting. No decoder changes.

A deterministic 2,048-clip scout subset is sampled from the original training pool, independently of any practice or exam outcome. Three independent degradation seeds (101, 202, 303). Short means a consonant occupies at most two 40-ms clean frames, not a short flattened string. Source word boundaries, token occurrence IDs, duration runs and blank-separated repeats are retained. Word-level reference duration spelling collapses within words only; repeated words and consonants across word boundaries do not collapse together. Teacher and clean student must agree exactly with the source word-derived token sequence before a clip is eligible. This is screening, not independent truth.

Selection requires the same consonant occurrence to be absent or changed in at least two of three degradations in an alignment allowing ±400 ms timing movement. Freeze at most 128 selected clips. Draw an independent random subset of the same provisional size from the same eligible pool, allowing and disclosing overlap. Both subsets undergo blind Soniox lexical transcript review. Reject mismatches; freeze equal verified counts using preregistered seed 810/811, without reference to training or benchmark outcomes. Upweight only those clips 3×, all other clips 1×. No short/repeat proxy upweights. The cap is 128/21,161 = 0.60% of the historical pool, unlike the previous 94.8% broad weighting.

Independent lexical review supports the as-said transcript but is not a fine pronunciation judgment. Store raw responses, audio hashes, source word text and phonemes, alignment positions and all degradation failures. No mined benchmark material. Held-out and gold exclusion remain active. Existing clean and studio performance, genuine mistakes, repeated-word/short-consonant behavior and self-corrections remain evaluation controls.

## Cached evidence diagnostic

Read-only replay of the original cached diagnostic incident/column set, preserving its 18 phone deletion pairs. Search 0, 120, 240 and 400 ms margins, retaining neighboring words in the alignment. `sparse-wide-full-evidence.json` contains each maximum posterior, time and aligned operation. In the exact word span, 14/18 are below .30; with ±400 ms, 12/18 remain below .30. The two added strong peaks are ل at 19.04 s and ي at 16.76 s in p32. Neither repairs the corresponding aligned deletion. The ي peak is earlier, not late. Repeated tokens already present elsewhere in a word are not recoveries. Top-five omission means below the retained top five, not zero full-posterior mass.

An initial narrower incident-window pass returned 15 pairs; its artifact is retained separately. Only the full original incident replay supplies the 18-pair comparison. No threshold was fitted or decoder changed.

## Objective audit

`modal/sparse_audit.py` uses the saved paired-control configuration to find the original initialization and audits three fixed batches without updating weights. The decisive projection-layer results and full-network limitations are recorded below. An initial missing-helper failure and exploratory identity-gradient failures are retained in the audit artifacts, not counted as passed checks.

## New self-correction controls

Stored scripted recordings contain two performed candidates previously omitted by the three-way control harness: rec-yazin-rec-v1-01-t1, 96:2:4 (طين then علق); rec-yazin-rec-v1-02-t1, 71:7:6 (وجعلوا then جعلوا). The first full take was rejected for other ayahs, so only the independently checked local correction is used. These are actual recorded attempts, not splices or repeated-phone proxies.

Blind Soniox stt-async-v5 transcriptions retain both attempts in each window. Exact model/config, token timestamps and raw responses are in `sparse-selfcorrection-soniox-raw.json`; readable results are in `sparse-selfcorrection-soniox.json`. Verification is independent-model evidence, not a new human listening verdict. Source audio remains evaluation-only. Replay retains every warning state, including an alarm later resolved, and records attempted/corrected phonemes separately. Other portions of the recordings are not newly asserted clean.

## Required result

Before training: resolve the objective audit and freeze verified equal-size manifests. Then train selected and random arms with exactly the same original budget. Export and compare final int8 SDK evidence on lab-v1.5 practice phone and studio only. Report all recovered/newly missed mistakes, removed/new warnings, short/repeated-word and self-correction outcomes, and comparison with original uniform adaptation. Record the selected arm through `tools/lab.py run`. No checkpoint selection, exam access, fold or discard.

## Mining preflight result

The first exact-emission-sequence filter rejected every candidate because it treated blank-separated acoustic duration emissions as separate source tokens. No model was updated. The corrected filter groups repeated emissions for one source-token occurrence while retaining their frame spans and multiplicity; it refuses ambiguous same-token word boundaries. Nine helper tests cover boundary preservation, repeat consumption, timing, caps, reproducibility and the verification gate.

The corrected frozen scout run has 162 eligible clips and 27 with a persistent consonant failure. All 27 are provisionally selected; 27 clips are independently sampled from the same eligible pool, with three overlaps. Independent transcript review is complete; final accepted counts are below. Both arms keep the same original training pool: JSON-serialized manifest SHA256 `ba66b2fb1c88ebf89fe260805cdc351f33bbb7347f577ef74ed819e06c057131` (newline-delimited keys hash `920613896955694b961d03661928e7bb45fa8b6aa0a9cd72acf9944293a4dd1d`). Model initialization is `spliced-clean`. Nothing has been selected using practice/exam results.

Projection-layer and full-network gradients are reported separately. Full-network repeated differentiation is affected by the model's custom backward machinery; disabling Balancer/Whiten, auxiliary losses, or replacing Swoosh activations in diagnostic copies did not establish equivalence. These ablations are not training changes, and no causal explanation of the old full-run regression is claimed from them. The original single-view objective remains the training recipe for both sparse arms.

## Frozen verification and corrected recipe

Blind Soniox review accepted 16/27 in each provisional arm under exact per-word lexical matching with existing Uthmani spelling variants; the rest are excluded from upweighting, not relabelled. This can reject an orthographic ASR variant as well as a real mismatch, intentionally favoring conservative inclusion. The final upweighted fraction is 16/21,161 (0.0756%), weight 3× in each arm, all other clips 1×. The equal-size manifests, source and independent-review hashes, and two independently verified self-correction controls are frozen in `sparse-verified-manifest.json`. No training target was rewritten. There are no repeated-word clips in the selected set; repeated words and actual corrections are controls, not claimed mining coverage.

The projection-layer audit (`sparse-projection-audit.json`) avoids the custom encoder-backward machinery. On three fixed batches, noisy CTC gradients have 0.75, 1.09, 0.91 times the norm of weighted KD, and clean CTC has 0.75, 1.03, 0.92 times. Clean CTC versus KD cosines are -0.79, -0.86, -0.59. Thus the added clean transcript term is a comparably sized, opposing force at the output layer, not a negligible addition; the paired control changed the intended balance. Averaging duplicate identical-input transcript terms gives exactly zero projection-gradient difference on all three batches. The single-view weighted loss also passes all-one CTC equivalence, all-one KD equivalence, manual-weight equivalence and finite-gradient checks on Modal.

This resolves the recipe choice for this experiment: restore the original single-view CTC1+KD2 for both arms; do not add or revisit consistency. It does not claim that an end-to-end ablation has assigned all of the earlier control regression to one cause. Full-network gradient decomposition remains confounded by custom gradient transformations and fresh-forward variability; those diagnostic ablations are preserved and do not modify the training model.

## Training launch provenance

The first two launches used wrapper defaults of six epochs and 3,000 diagnostic examples. Both were stopped before any `data`, `adapt`, or `train` log event and before optimizer updates. The corrected launches use the historical one-epoch cap and 300-example diagnostics, with the same 200-update limit. Source commit: `4b0b6b8`. Modal apps: selected `ap-SPQJOfTV8utDf1zyZv5Odu`, random `ap-9VjR7penz79yW54blVXL7P`. Full data/config and completion records will be checked before evaluation.

The baseline passes both verified self-correction controls in free and declared modes. Tracker positions are locked around the correction in all four runs; no target warning was ever confirmed. `sparse-selfcorrection-baseline-v2.json` preserves positions, all warning states and raw phoneme evidence.

## Training result

Both arms completed exactly 200 updates with finite reported losses and zero skipped updates. They used the same 21,161 clips, 50.000777 hours, initialization at step 30,287, and matching configuration apart from run name and sparse arm. Selected diagnostic PER versus reference is 9.05264%, random 9.04079% (300 gold-excluded clean-match test clips). These diagnostics do not select checkpoints. Both scheduled final checkpoints are retained. Final frozen subsets overlap on two clips.

## Exposure correction before practice results

The read-only schedule audit found 236 planned batches; the fixed 200-update budget consumes 17,360 clips. Only 12/16 selected clips and 15/16 random clips appeared in that prefix. Although the nominal manifests matched, effective weighted exposure did not. V1 is therefore superseded and cannot answer the selection question. No practice score was read. Its in-flight decoding and controls were stopped; the completed selected self-correction replay is retained as a superseded diagnostic only.

V2 freezes 12 verified clips in each arm, all present in the same unchanged 200-batch prefix. The selected arm retains all 12 exposed qualifying clips; the random arm takes a seeded subset of its 15 exposed verified random clips (seed 811). This conditions both arms on exposure without changing the original pool, order, weights, augmentation, loss, or budget. `sparse-verified-manifest-v2.json` and `sparse-exposure.json` record the change. Training now fails before any update if a frozen upweighted clip is outside the budget. Both v2 runs restart from the original `spliced-clean` checkpoint; no v1 checkpoint is reused. Distinct model/evidence names prevent cached v1 outputs from entering v2 evaluation.
