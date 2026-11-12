# Local recorded-background contrast training

Status: training implementation prepared; blocked before real-data training and export.

## Prediction and decision rule

The card predicts listener −1 and overall −1 on practice, with one false warning removed and no predicted losses. The proposed cause is a phone head associating consonants with recording background instead of speech identity.

The three registered targets are:

- `lab-v1.5-p11/phone/warning:wrong_word/26:106:1/104.32`
- `lab-v1.5-p36/phone/warning:wrong_word/55:48:1/282.64`
- `realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`

Compare the training-selected crossed head with the identical-view ordinary CTC head on actual engine-requested evidence. The task's final instructions require at least three target wins to proceed to practice scoring, even though the card's narrower kill criterion says zero. Neither threshold has been evaluated. Target wins are **unmeasured**, not zero. Practice scores used: **0 of 3**.

## Implemented mechanism

`modal/student_local_noise_head.py` provides separate `prepare`, `fit`, and `export` stages. Preparation reads help-train only. Whole speakers with any non-training split are excluded; a deterministic speaker-grouped partition reserves training speakers for validation. Only none-labelled, scored takes without mistake, repeat or unsupported annotations receive canonical transcript CTC supervision. Other authorized takes supply preservation KL to the shipped listener without being relabelled as correct.

Preparation requires a contiguous nonzero quiet recording segment of at least 240 ms, below 1% of the recording's 80th-percentile block energy. It pairs two speech recordings with one independently recorded background donor within the same partition. The unchanged controls and the background-swapped views are cached identically for both training arms, with waveform provenance, noise energies, speaker partitions and file hashes. The plan cannot be silently reused with changed features or pairing identities.

The waveform decomposition estimates the original background from the recording's quiet-segment spectrum. The control recomposes the original waveform; the swapped view retains exactly the same **estimated speech** and substitutes tiled donor samples at the estimated original background energy. This is a training augmentation, with no inference-time denoising.

`modal/student_noise_features.py` captures the shipped streaming encoder's frozen head inputs, including the actual dequantized activations consumed by its integer projection. It preserves streaming state and drops final padded output rows using the SDK's output geometry. Transcript-occurrence CTC forward/backward occupancies integrate over possible phone placements, including repeated labels; checker visits and hard word boundaries are not used.

All arms start from the same shipped linear head and receive identical views, batches, preservation examples and update counts. The ordinary arm uses length-normalized transcript CTC plus KL preservation. The crossed arms additionally require the same consonant occurrence to agree across its two backgrounds and different consonants under a shared donor background to remain distinguishable. Same-phone negatives are excluded. Pairing, 24 epochs, learning rate, KL weight, temperature and two candidate contrast weights are fixed in the recipe. Checkpoint and contrast-weight selection use mean CTC-plus-preservation loss across reserved training speakers, before any target access.

Export replaces only phone-head constants within the existing primary ONNX graph, retains quantization geometry, and checks output shape and encoder cache parity. It exports both control and crossed bundles. Every inherited SDK auxiliary file must match its registry size and SHA before export, and is retained in each bundle. Alphabet, streaming clock, tracker, checker and confirmation rules have no runtime edits. This exporter has not yet been exercised on the installed real model.

## Validation and infrastructure result

Six initial synthetic checks passed under the matching local Python architecture: exact control recomposition, donor energy matching, contiguous recorded-background selection, none-label and whole-speaker exclusion, deterministic partition-safe pairing, exhaustive CTC path agreement including repeats, and rejection of impossible complete transcripts. The six-test invocation combines the first two background properties in one test. Python syntax compilation also passed.

All **15 expanded synthetic tests passed** in an isolated native local Python environment: 10 background, occurrence, contrast and tiny-training checks, plus five ONNX feature/export checks. These additionally cover frozen donor-identity rejection, invalid occurrence inputs, collapsed/shared-background phone predictions, float and int8 head reconstruction, head-only export, per-channel quantization, final-row trimming and state carry, and complete inherited-asset requirements. The two-epoch training fixture learns and saves finite, checksum-bound selected heads; it uses six-frame synthetic sequences, not real training data. The changed contrast and training tests were rerun after fixture cleanup and passed. Commands were `python -m unittest test_student_local_noise_head` and `python -m unittest test_student_noise_features` from `modal/`, with CPU thread limits. This verifies the algorithm and fixture bundles, not the installed real model or Linux training dependencies.

Supervisor Modal requests **1–4 all failed** before any command output with `NotFoundError: No such file or directory`. Request 2 was the required six-shard `bash tools/autolab_tests.sh`; request 3 was a minimal `pwd`, filesystem inventory and Python-version command. Request 4 retried the Python suites using the training image. Each returned USD 0 and no command log. This establishes a runner blocker; it does not establish a training-data premise failure or an experimental kill. The core/SDK tests have not been verified.

No help-train inventory, real feature cache, selected checkpoint, candidate model bundle, or target replay was produced. No `.autolab/model.json` is declared: decoding with the main model would not evaluate the implementation. No killed outcome is declared, because neither the premise nor target gains have been measured.

## Limits and work required

Raw quiet-segment energy cannot guarantee the segment contains only background; quiet breaths may pass. Speech is estimated rather than independently isolated, and separation may remove weak speech or leave original noise. Donor tiling can introduce artificial periodicity. Those limitations could defeat the intended speech/background crossing even if the algebra and energy checks pass. Frozen encoder features and preservation KL limit drift but do not establish preservation of real substitutions, clean sessions, or studio behavior.

After the supervisor runner is repaired, inventory the available training labels and complete installed parent bundle. Run `prepare --model <complete-parent-directory> --out /scratch/listener-local-noise-contrast-training/training`, then `fit --out <same-directory>`, then `export --model <same-parent-directory> --out <same-directory>` through the supervisor's training image. These stages freeze selection before target evaluation. Verify the complete bundles and compare both heads on the registered targets through the ordinary SDK audio/evidence route, including inherited auxiliary behavior. Fewer than three target wins requires reverting the implementation and recording the measured kill; three wins requires a practice score before a gate decision. This experiment cannot presently be handed off as a trained or scoreable listener change.
