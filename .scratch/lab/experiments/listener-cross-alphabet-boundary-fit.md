# Listener cross-alphabet boundary fit

## Preregistered prediction

Experiment `listener-cross-alphabet-boundary-fit`, listener, explore. Phone-only
supervision may assign an extra word's consonants to neighboring word intervals.
Joint acoustic-phone and compositional Arabic-letter fitting with shared latent
word boundaries predicts listener **−1**, overall **−1**, with no losses predicted.
That forecast is below the fold threshold. The experiment card is unchanged.

## Mechanism implemented

`modal/student_boundary.py` implements a training-only synchronous word lattice.
For a recording with T frames and W ordered transcript words, a latent partition
chooses W−1 cuts from the T−1 internal frame edges. Every word interval is
nonempty. Each alphabet uses its own blank-interleaved CTC state machine within
the interval; the phone and letter emissions need not occur on the same frames.
The joint lattice changes words only when both machines reach terminal states.
Across a boundary, either alphabet still requires a blank between equal adjacent
labels, so resetting a word cannot invent an articulation lost by the exported
CTC collapse. Leading/trailing blanks and real pauses remain legal alignment
states rather than forced speech.

The shared arm sums products of the two alignment likelihoods over the same
partition. The independent control takes the product of the two separate
partition sums. Both use uniform priors over the same admissible nonempty
partitions. The dynamic program marginalizes the boundaries and within-word
alignments exactly; it does not force-align the recording, detach the timing
posterior, fit boundary constants, or add an inference separator head.

`modal/student_boundary_fit.py` admits only supplied training-speaker none takes
without error events, repeats, unsupported annotations or exclusions. It can also
admit an explicitly enrolled catalog of authorized, verified performed
word transcripts with matching audio hashes. Any speaker with a non-training
record in either corpus is excluded before admission. Optional explicit speaker
exclusions are applied in the same pass. The supplied help-train volume's
release-speaker exclusion remains a required upstream guarantee; that guarantee
could not be checked because the worker never executed.

For admitted none takes, connected phone words come from the reference's word
offsets, with duration runs collapsed within words. Arabic spelling removes
diacritics, tatweel and Quranic modifier-letter pronunciation guides, normalizes
alif wasla, and preserves hamza and ordinary compositional letters. Additional
training transcripts must provide both alphabets for every word; mistake
scripts are never treated as performed transcripts.

The exact shipped streaming ONNX encoder is frozen and exposes its final
activation on full original recordings. No new augmentation family is added.
Original ordinary speech and real pause frames remain in the cache and in the
shipped-output KL preservation objective. A phone affine head starts from the
shipped dequantized readout. The training-only letter affine head starts from
corresponding phone rows where available, with deterministic initialization for
orthographic-only letters. Both arms receive identical copies, cached features,
recording order, optimizer, clipping, preservation objective and update count.
Both compute both marginal sums and both gradients per update, discarding the
unused gradient to match compute while applying only the arm's own objective.

Defaults are six epochs, Adam at 1e−4, gradient norm clipping at 5, and a maximum
20-hour budget including caching. These choices are fixed before any practice
inspection. One common epoch is selected by the mean of the arms' independent
boundary NLL plus posterior preservation, averaged by speaker on a deterministic
20% validation partition of admitted training speakers. Practice readers,
incidents and labels never participate in training or checkpoint selection.

`modal/student_boundary_export.py` reuses the native frozen-encoder export
contract from `exp/listener-implicit-sequence-head-fit`. It extracts the native
affine head, including dynamic int8 weights; removes the replaced head; preserves
encoder/state graph, ordered alphabet, front end and output clock; and copies
the existing insertion teacher and all external weights with checksum checks.
Only the fitted phone head is exported. The fitter requires optimized-runtime
parity on every cached real recording and complete teacher assets before marking
either bundle verified. SDK decoding, tracker, checker and insertion verification
code are untouched.

## Reproduction on an available authorized worker

```sh
python3 modal/student_boundary_fit.py cache \
  --model /lab/data/models/spliced-clean-int8 \
  --help-train /lab/data/help-train \
  --out /scratch/listener-cross-alphabet-boundary-fit/cache
python3 modal/student_boundary_fit.py fit \
  --cache /scratch/listener-cross-alphabet-boundary-fit/cache \
  --out /scratch/listener-cross-alphabet-boundary-fit/fit
```

Optional `--corpus-index` and `--exclusions` must name authorized training-only
inputs. The fitter produces `fit/shared/` and `fit/independent/`, each with
`model.onnx`, a new manifest with its checksum, unchanged insertion-teacher assets,
and a verification report. Neither bundle should be declared in
`.autolab/model.json` until both finish verification. The script requires Torch,
NumPy, ONNX, ONNX Runtime, torchaudio and soundfile on the worker.

## Validation and limits

All **19 synthetic unit tests pass**, without skips:

```sh
.scratch/boundary-venv/bin/python -m unittest discover -s modal \
  -p 'test_student_boundary*.py' -v
```

Only tiny generated numerical examples and ONNX graphs ran on the Mac. The
local test environment used Torch 2.15.0, NumPy 2.5.4, ONNX 1.23.2 and ONNX
Runtime 1.31.1. The
suite checks both exact marginals against exhaustive frame-path/partition
enumeration, single-word equivalence to ordinary CTC, different within-word
timings, incompatible boundary evidence, repeated articulations within/across
words, finite-difference gradients for both objectives, unreachable-state
gradients, rejection of impossible transcripts, and use of only the chosen
gradient despite matched forward/backward compute. Admission tests cover Arabic
spelling, word-local duration collapse, repeat/error rejection, cross-corpus
speaker exclusions, audio identity and root containment. Export tests cover
float/int8 readout extraction, unchanged state output, removal of the replaced
head, a complete generated external-weight teacher bundle, source immutability,
manifest/clock preservation and missing-asset rejection.

Python compilation, runner CLI parsing and whitespace checks also pass. These synthetic checks do
not establish compatibility or parity with the actual shipped model; the fitter
still requires real-recording export verification before declaring bundles.

All four authorized supervisor requests failed before command execution:

1. Run 1, supplied-data/model/dependency inventory: failed before execution with
   `NotFoundError: No such file or directory`.
2. Run 2, numerical boundary tests: the same pre-execution failure.
3. Run 3, the prescribed six-shard core/SDK build and suites: the same
   pre-execution failure.
4. Run 4, retry with the complete numerical/export suite and data inventory:
   the same pre-execution failure.

There is no inspected real source bundle, encoder cache, trained head or verified
export. Core/SDK results, full-session decoding, target consonant recoveries,
target warning wins, collateral losses and practice deltas are **unknown**.
No practice-score calls have been made. No `.autolab/model.json` is declared;
running the main listener would not evaluate this implementation.

This is an infrastructure block, not an absent acoustic premise or a measured
zero gain. The engine implementation is retained for resumption. No killed
outcome, fold or discard is asserted.

## Remaining decision

After both real bundles pass verification, compare their complete exported SDK
session decoding on the five card targets. Record whether shared fitting recovers
a previously missing, correctly ordered planted consonant relative to the
independent-boundary control, as well as actual target warning gains. Do not
select another checkpoint or constants from these observations.

The user's stopping rule requires at least three targets won before requesting
practice scoring, despite the smaller preregistered forecast. If the premise
fails or fewer than three are won, revert engine changes, commit the measured
negative write-up and write the requested killed outcome. If three or more are
won, request the authorized practice score before deciding or narrowing; the
fold gate trades collateral losses against wins. Ordinary-reading and real-pause
preservation may limit drift but do not establish loss-free behavior.
