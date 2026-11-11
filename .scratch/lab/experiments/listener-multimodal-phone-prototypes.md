# Multimodal acoustic phone prototypes

Status: blocked by supervised Modal execution; acoustic premise untested.

## Prediction

The card predicts two false warnings removed and one caught mistake lost:
listener −1, overall −1. The target cluster is listener/wrong_word/false-warning.
The task's continuation rule requires at least three target recoveries before a
practice score, despite the card's lower two-window kill threshold. Neither
threshold has been evaluated. No practice score has been requested.

## Mechanism and implementation

`modal/student_prototypes.py` implements a normalized mixture of spherical
acoustic prototypes for each phone, including blank. Each prototype scores a
normalized pre-head encoder vector. Log mixture weights are normalized within
each phone; log-sum-exp marginalizes prototype identity before the existing
phone softmax. Duplicating a prototype therefore does not increase phone mass.
The CTC alphabet, blank identity, output shape and 40 ms frame clock are unchanged.

Initialization uses forward-backward occupancies conditioned on complete
transcripts, including CTC's blank separation for repeated phones. Weighted
spherical clustering assigns acoustic centers from these occupancies rather
than from the shipped head's winning labels. This can retain legitimate
renditions the original head currently misclassifies. It does not encode word,
verse or reader exceptions.

`modal/student_prototype_train.py` defines training for two and four prototypes,
with a conventional linear-plus-residual-MLP control for each parameter budget.
For the lite model's 256-dimensional, 43-class head, the two-prototype head has
22,146 parameters and its control 22,151. The four-prototype head has 44,248 and
its control 44,351. The control starts at the original linear classifier's
outputs. Both families train their head and last encoder stack on identical
batches and seeded draws from the existing audio/spec augmentation code.

The utility reads original clean-match training recordings and help-train
training speakers' none-labelled takes only. It excludes whole gold/silver
release speakers using their existing metadata, and rejects help recordings
carrying any mistake or repeat event. It uses the shipped transcript compiler,
CTC, available original teacher KD, and KL preservation against the parent
model's outputs. Hashing speaker identities creates disjoint training and
checkpoint-selection partitions. These metadata assumptions still require the
remote input audit; no recording has actually been loaded here.

Model family size and checkpoints are selected exclusively on the frozen
training-speaker selection partition. The matched conventional control uses
the same prototype parameter budget as the selected architecture. The selected
model is frozen before any practice-window replay.

`modal/student_prototype_export.py` exports the selected primary model through
the existing streaming ONNX wrapper and int8 MatMul recipe. It checks fp32 ONNX
against an explicit cached Torch stream on training clips, checks frame/output
shape parity and reports int8 drift. The existing SDK directory loader and
listener can consume this interface without any tracker, checker or
confirmation-rule edits. No trained checkpoint or exported bundle exists yet.

## Validation and limits

The added Python unit tests cover normalized marginalization, duplicate-prototype
invariance, separated legitimate class realizations, finite gradients, control
initialization and parameter count, impossible transcripts, and occupancies
against exhaustive CTC paths including repeated labels and an empty transcript.
They have not executed successfully: the remote requests failed before returning
test logs. Local Python compilation and `git diff --check` pass.

Supervised requests failed with `NotFoundError: No such file or directory`:

- Run 1: L4 training-image input/checkpoint/storage audit, five-minute reservation.
- Run 2: CPU training-image prototype unit tests, five-minute reservation.
- Run 4: prescribed six-shard core/SDK build and test harness, fifteen-minute reservation.
- Run 5: minimal `python3 -c 'print("container reached")'` execution,
  two-minute reservation, also failed with the same missing-resource error.

Run 3 retries the Python tests in the L4 training image; its result is pending.
The minimal run 5 failure establishes that the blocker does not depend on
training imports, dataset paths or the unit-test command. The failed audit returned no checkpoint inventory, training
schema, speaker counts or container storage measurements. The supervisor
reported $0.24 spent after run 1; no training was performed.

This is an infrastructure blocker, not a negative acoustic result. Target
recoveries, complete written-versus-rival preferences, full-session streaming
behavior, clean-session warnings, studio behavior and aggregate wrong calls are
all unmeasured. No target gains or zero-loss claim can be inferred from the
mathematical tests or source inspection. Primary-model drift remains the
forecast's central risk.

The experiment card, supplied manifests, evidence, test data, scorer and lab
tooling are untouched. No model selection override, fold or discard has been
performed. Do not score this source change with the main listener as a substitute
for the missing trained prototype model. Completion requires restored supervised
execution, a verified parent checkpoint, training-only selection, a complete
export, actual SDK full-session target replay, and the task's conditional
practice score.
