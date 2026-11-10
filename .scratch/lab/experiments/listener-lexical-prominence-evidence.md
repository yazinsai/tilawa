# Lexical prominence evidence

Status: ready-for-human — supervisor execution is unavailable; acoustic premise unmeasured.

## Prediction and mechanism

The unchanged experiment card predicts one false substitution warning removed,
no losses counted: listener −1 and overall −1. The hypothesis is that consonant
blur can leave relative syllable prominence intact even when the existing
listener prefers an incorrect pronunciation.

The proposed CPU scorer compares the complete written and already-accused
pronunciations on the identical existing string-path acoustic window. Its
observations are relative vowel duration, energy and pitch movement. Syllable
weight and stop-versus-connected position condition a learned prominence
pattern; they do not impose a deterministic stress rule. Syllable boundaries
and recitation lengthening must be marginalized. Fit and model selection must
use only none-labelled help-train training speakers, with speaker-grouped
training-only partitions and release-held-out speakers excluded.

A veto requires both different predicted prominence patterns and a frozen
acoustic preference for the written pattern. Primary decoding, tracker events,
parallel-graph confirmations and insertion decisions must remain unchanged.
The task's continuation requirement is at least three actual target wins. The
card's smaller prediction does not waive that requirement.

## Work completed and limits

A temporary engine draft exposed read-only discovery of complete, otherwise
supported string-path hypotheses at the existing sequence-scoring windows,
including word-local and paired paths. It kept the complete sequences and
marked the accused word's range within their identical neighbour context.
Graph-path comparisons bypassed the hook. The draft reused the checker's
existing labelled-pause test for stop conditioning.

A temporary SDK syllable compiler grouped expanded madd labels into one
nucleus, kept consonant gemination as coda/onset weight, and centered the
conditioning design inside each word. Identical weight/position designs
abstained even when the consonants differed. This is a structural prerequisite,
not a trained predictor or evidence about any target.

Lightweight local checks passed for vowel lengthening, separate syllables,
gemination, stop conditioning and identical-design abstention. A synthetic
checker replay also verified that discovery returned the complete original
window without changing the baseline verdict, and that a false callback could
veto its string-path substitution. These checks used Node's TypeScript
transformation and temporary copies with import suffixes adapted. They were
not Vitest, a TypeScript build, an acoustic evaluation or an independent
listening judgment.

The incomplete engine draft is retained under ignored
`.autolab/prominence-draft/` for resumption. It is removed from the branch's
engine diff. No trained scorer, registry descriptor or model bundle is shipped.

## Execution blocker

The authorized supervisor failed before container execution. Requests and
results are retained under `.autolab/modal-request-*.json` and
`.autolab/modal-result-*.json`:

1. Run 1: read-only authorized training-data and model-layout inspection.
   Returned `NotFoundError: No such file or directory`.
2. Run 2: the required six-shard `bash tools/autolab_tests.sh` harness.
   Returned the same error.
3. Run 3: minimal working-directory and `/lab/data` inspection.
   Returned the same error.

All three failure results report zero compute cost and supply no container
logs. They do not establish missing training recordings, missing prominence
differences, or a fired kill criterion. No training recording or target window
was accessed. No model was fitted or selected. Target wins and the overall
change are unknown; zero practice scores were requested, so all three scoring
attempts remain available. No killed outcome is asserted.

Restore supervisor execution before continuing. Inspect the authorized data
and split assignments, obtain actual complete target comparisons from the
unchanged engine, and test whether their predictor inputs can differ. Fit and
freeze the acoustic model on training speakers before target acoustic
evaluation. If fewer than three targets are won, record the premise kill; if
at least three are won, complete the side-asset bundle, preserve the existing
primary and teacher assets, pass the required tests and run practice scoring.

Variable recitation style, uncertain vowel boundaries, sparse syllable-weight
coverage and unreliable phone pitch estimation remain untested risks. A
positive prominence preference can still forgive a genuine substitution;
aggregate losses belong to the fold gate after the required target validation.
No scorer, lab tooling, supplied manifest, evidence, test data or experiment
card was changed. No fold or discard was performed.
