# Crossed-context frozen temporal readout

Status: needs-info. The implementation draft builds, but the authorized Modal
supervisor fails before command execution. No acoustic result or complete fitted
bundle is available. The experiment card is unchanged.

## Prediction and mechanism

The card predicts one false warning removed, no losses, listener −1 and overall
−1. Its premise is that the consonant readout sometimes follows surrounding
sounds instead of the disputed interval. This remains unmeasured.

`modal/student_crossed_export.py` exposes the frozen pre-CTC encoder features in
a separate ONNX graph. Export compares every original output and cache on eight
consecutive streaming windows with optimizations disabled. It copies the primary
graph, manifest and existing auxiliary files unchanged. A seeded 32-dimensional
projection feeds three identical 48-unit tanh heads with seven-frame temporal
inputs, including identical zero padding.

Only help-train training speakers' none-labelled takes supply supervision. A
speaker appearing in any non-training split is excluded entirely; explicitly
excluded takes are also skipped. Exact CTC forward/backward provides soft phone
occupancies. Complete legal paths are sampled from that posterior each epoch to
retain uncertain boundaries, including mandatory blanks between repeated phones.

The crossed arm constructs two operations over two-phone flanks and a central
consonant interval. Context exchange takes both flanks from another recording
with the same consonant and different adjacent phones, retaining the original
central features and labels. Central exchange takes a different consonant from
another recording with matching adjacent phones, retaining the source flanks.
Every composed frame is supervised with the sampled phone/blank label of the
interval actually used. In particular, a new central consonant receives its own
label rather than the original transcript's label.

All arms share initialization, capacity, optimization, original soft supervision
and update counts. Half the training loss remains unmodified real examples. The
ordinary arm's other half uses additional unmodified posterior-path samples. The
unstructured arm uses the same two composition operations without phone/context
matching, with correctly reassigned labels and equal classified-frame counts.
Recording-balanced sampling prevents long takes dominating training.

Three speaker folds, entirely within training speakers, select 6, 12 or 18
epochs. Regularization and geometry are fixed before acoustic evaluation.
Selection first minimizes the fraction of actual scripted-substitution engine
comparisons newly vetoed, then speaker-averaged soft-label loss. Script locations
only identify preservation requests; no planted word is assumed to have been
performed. Missing preservation coverage or either crossed operation aborts
training instead of producing an apparently validated bundle.

## Listener integration and premise diagnostic

The SDK compares both complete pronunciations over the checker's existing
string-path support windows. All three heads marginalize CTC boundaries under
the same completion-conditioned timing prior. A certificate requires strictly
finite disagreement: crossed prefers written, ordinary and unstructured mixing
both prefer rival. Ties, unknown phones and impossible pronunciations abstain.
Certificates name exact pronunciations, word occurrences and native frame bounds.

Only the public wrong-word verdict is downgraded. The checker's internal
confirmation and ownership ledger remains intact. Primary phones, posteriors,
tracker inputs, graph confirmations and insertion decisions are preserved.
Audio sessions publish matching downgrades at end and store certificates for
evidence replay. The Node loader rejects incomplete or corrupt declared assets,
checks the primary hash, inventory and dimensions, and releases the separate
feature backend with the primary backend.

`modal/student_crossed_probe.py` is a read-only premise diagnostic on the SDK's
actual requested windows. It samples 16 written-transcript-conditioned paths,
holds each disputed raw central interval exactly fixed, and replaces its
immediate readout context with same-consonant, different-context intervals from
up to 16 training recordings. Donors must cover at least three speakers. A
changed consonant is compared with its minimum-edit rival, or blank for a
deletion. The diagnostic reports original and intervention central log-odds;
its robust sensitivity summary requires a donor-median reversal on a majority
of boundary draws and an ordinary complete-pronunciation rival preference.
These diagnostic outputs never select a model or training constant. This probes
the temporal readout's context dependence, not the encoder's entire receptive
field. Central features can already contain surrounding information; the
intervention cannot remove that, and artificial joins may not transfer.

## Validation and outstanding work

Both core and SDK TypeScript builds pass. The complete SDK suite passes 34 tests with one
optional help integration skipped. All nine Python mechanism tests pass,
including a single-update, synthetic-data smoke check of all three Torch heads. Checks include exact soft
occupancies against enumeration, complete posterior path sampling, correctly
changed central labels, cross-recording donor constraints, unchanged disputed
features, temporal padding and complete-pronunciation timing normalization.
All six new core certificate tests pass with one worker, including actual
insertion preservation and graph exemptions.

The initial parallel core run timed out in numerous tests and reported worker
RPC timeouts. It also caught two end-of-Quran safeguards lost during adaptation
from the older integration; both original safeguards were restored. The full
single-worker rerun (`--testTimeout=60000`, retaining explicit case limits)
passed 771 of 772 tests. One startup-persistence case took 16.8 seconds against
its existing 15-second limit; it also timed out in isolation and with the
unchanged HEAD checker. The run reported two worker RPC deadline errors of the
kind documented by `tools/autolab_tests.sh`. No timeout or existing assertion
was changed. All ten end-of-Quran tests pass after restoring the guards. Logs
are retained untracked under `.autolab/local-tests/`. Thus the complete core
suite and the requested Modal harness have not passed, despite all new
mechanism and integration tests passing.

Supervisor requests 1, 2 and 3 all returned
`NotFoundError: No such file or directory` before executing their commands.
They attempted an inventory, the supplied six-shard test harness, and the
test/extract/train/export job. Thus help-train coverage, real preservation
comparisons, fitted heads, ONNX/cache parity, normal fitted SDK loading, context
sensitivity and target removals remain unmeasured. No practice scores were used.
No `.autolab/model.json` or killed outcome is declared: infrastructure failure
does not establish a premise failure or zero wins, and the primary model alone
would not evaluate this change.

The complete export job, after the supervisor is restored, is:

```sh
python3 modal/student_crossed_export.py \
  --model /lab/data/models/spliced-clean-int8 \
  --out /scratch/listener-crossed-context-readout/bundle \
  --labels /lab/data/help-train/labels.jsonl
```

Then use `packages/sdk/src/bench/crossed-context-cli.ts` with the verified bundle,
audio and existing evidence to inspect actual warning windows and removals; pass
those diagnostic files to `student_crossed_probe.py`. Fewer than two sensitive
targets fails the card's premise. Fewer than three actual target removals calls
for rollback and a measured kill under the user's stopping rule. Three or more
removals requires practice scoring before deciding. No such decision is yet
supported by observations.

This draft buffers session audio and performs a separate feature pass at end,
so memory and recheck latency grow with session duration. It does not establish
immediate streaming precision or mobile deployment. Small-data alignments and
artificial feature joins can suppress real substitutions despite the controls.
The proposed bundle must be completed and verified before scoring or folding.
No supplied data, evidence, scorer, lab tooling or card was edited; no fold was
performed.
