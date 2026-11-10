# Listener routed acoustic specialists

Status: blocked, unmeasured. The source prototype is implemented, but no training, executable tests, model export, full-session target comparison or practice score completed. There is no declared replacement model. A decode with the main listener would not evaluate this experiment.

## Preregistered prediction

The card predicts listener −2 and overall −2 wrong calls: three recovered added words with one collateral wrong call. The proposed cause is conflicting upper-layer gradients from short consonants, sustained voiced sounds and silence. Conditional parameter specialization should help beyond an equally sized dense block.

The seven registered targets are:

- `lab-v1.5-p07/phone/mistake:added_word/37:36:1/144.12`
- `lab-v1.5-p08/phone/mistake:added_word/37:102:16/72.17`
- `lab-v1.5-p33/phone/mistake:added_word/26:208:1/156.16`
- `lab-v1.5-p48/phone/mistake:added_word/12:76:1/39.98`
- `lab-v1.5-p56/phone/mistake:added_word/32:21:1/106.76`
- `lab-v1.5-p57/phone/mistake:added_word/51:18:1/95.26`
- `lab-v1.5-p57/phone/mistake:added_word/51:20:1/112.26`

The card's stated consonant-recovery criterion is relative to both shipped and the matched dense control. The later execution instruction requires at least three target wins before practice scoring. Neither criterion has been measured; an execution-service error is not evidence for a scientific kill.

## Implemented mechanism

`modal/student_specialists.py` replaces `feed_forward3` in the final Zipformer encoder layer with three copied feed-forward experts and a linear, softmax router. The experts start with identical shipped weights, and the router starts uniform. At inference the input is the acoustic activation alone. There is one ordinary CTC output head; the surrounding residual connection, attention and convolution caches, inventory and output clock are retained.

The last internal encoder is downsampled by two relative to the 20 ms embedded features, so its routing activations already share the 40 ms CTC clock. Supervision uses that direct alignment and rejects a different clock. Pause symbols and CTC blank map to the silence regime, vowels/glides/nasals to sustained, and other symbols, including qalqala, to transient. Exact CTC forward/backward occupancy is conditioned on collapsed authorized reference phones, with no teacher-decoded edge phones in the router targets. Impossible alignments abstain. The router loss masks padding and balances regimes by their occupancy mass.

The dense control expands the same block to three concatenated hidden groups plus two channels, with scalar input and per-output gains to exactly match the router/expert parameter budget. It also reproduces the shipped function at initialization. Its extra channels are connected trainable capacity. It has no routing input.

`modal/student_specialist_train.py` supplies matched continuation and export paths. The fixed, unevaluated defaults are seed 17, eight hours of clean-match Quran training clips, three joint epochs, 200 padded audio seconds per batch, AdamW at `1e-4` with weight decay `0.01`, shipped-posterior KL weight 4, and balanced routing loss weight `0.25`. The routed arm gets 100 router-only warm-up steps with experts frozen. Both arms otherwise use the same data order and augmentation draws, and update only the replacement block. The lower encoder and ordinary CTC head are frozen.

Quran clips use the existing gold exclusions. Help supervision admits only training-speaker, none-labelled takes without exclusions, relabels, errors, repeats or unsupported events. Cross-split speaker overlap and unsafe audio paths fail explicitly. Every fifth eligible help speaker and every tenth Quran training clip are reserved for training-only checkpoint selection. Validation uses fixed augmentation draws across epochs. Shipped noise/RIR banks, gain and SpecAugment are retained, as are acoustic CTC targets with shipped edge handling and cached teacher KD with its configured delay. Both models use chunk 16 and left context 128.

The export path creates the ordinary streaming fp32 graph and dynamic MatMul int8 variant. It checks fp32 outputs and caches against the same PyTorch streaming call loop, checks int8 frame shape and finite posteriors, and reports quantization disagreement on held-back training audio. It requires and checks copies of `teacher.onnx` and `teacher.onnx.data`, which the unchanged SDK uses for insertion verification. Existing model loading reconstructs the checkpoint's optional block type. No tracker, checker, warning rule, SDK inference path or existing manifest was edited.

## What was validated

Local `python3 -m py_compile` passed for the five changed/new Python files. `git diff --check` passed. Review against the pinned icefall source established that the replacement is called in normal and streaming paths and that the final internal block has the same clock as CTC. These are source/static checks, not execution evidence.

Ten unit tests were written for arbitrary-router identity, exact dense parameter equality, gradients, padding and clock guards, exhaustive CTC path occupancy (including repeats and impossible alignment), symbol regimes, deployed cache identity and full-encoder ONNX parity, isolated nontrivial audio-only ONNX routing, training-speaker/data restrictions, matched augmentation/fixed validation, and mandatory verification assets. They have not executed.

Five supplied supervisor requests failed with `NotFoundError: No such file or directory`, with no container logs or test results returned:

1. Training-image CPU discovery of shipped checkpoints, data and Zipformer implementation.
2. Required six-shard `bash tools/autolab_tests.sh`.
3. A minimal H100 training-image Torch/CUDA probe.
4. Training-image Python unit tests with ONNX dependencies installed and an explicit output directory.
5. A further training-image Python unit-test request during the resumed implementation review.

No GPU training or export ran. The supervisor reported approximately $0.12 charged and $139.88 remaining. No practice-score calls were used. Target recovery, overall change, component attribution, clean-session behavior, studio drift and the reader-removal gate are all unknown.

## Limits and work remaining

The training driver is an unexecuted prototype. The shipped checkpoint location/configuration and supplied dataset contents could not be confirmed. Identity and parameter equality are designed and tested in source, not established by passing tests. CTC blank is a proxy for silence and can also occur inside speech; occupancy supervision does not independently establish acoustic-regime separation. Restricting specialization to one feed-forward residual block may be too small an intervention. Global model replacement can still cause collateral drift even with frozen lower layers and preservation loss.

Resume by running the Python suite on the supplied training image and the required SDK/core suite, then confirming the shipped checkpoint. Train routed and dense arms with identical CLI settings and export/verify complete bundles. Decode the registered targets through full sessions using the ordinary SDK and compare previously missing planted consonants against both shipped and dense. If the scientific recovery criterion fires, revert the engine prototype and record a killed outcome. If at least three targets recover, declare the verified routed bundle in untracked `.autolab/model.json` and run practice scoring before any gate-based narrowing. The supervisor's default-model fallback must not be recorded as an evaluation of routed specialists.

The experiment card is untouched. The prototype remains on the experiment branch; this record does not claim completion or reject the acoustic premise.
