# Pause-mask ear agreement for wrong words

Status: engine implementation committed and both packages build and pass their
unit suites. Acoustic validation awaits the exact saved pause-mask export in an
authorized input mount. The experiment card is unchanged.

## Prediction

Keep the shipped decode and ask the existing pause-mask-trained student to agree
with each string-path wrong-word accusation. Predict three of four target
warnings vetoed, with about one caught swap lost: listener −2, overall −1.
The registered kill condition is fewer than three target accusations vetoed by
the built unchanged-checker rejudgment. Missing weights do not establish that
condition; no zero-win acoustic result is claimed.

## Mechanism

The SDK registry declares the existing final 800-update int8 export, SHA256
`22c8e197f2e60955f0b7794a8198db4d5ae9ee57aa679d3fc163075092fd8c58`,
72,403,248 bytes. Its loader validates inventory, streaming geometry, state layout,
mask clock, byte length and hash. The primary manifest stays byte-identical;
no auxiliary selector is added to it. No weights are retrained or substituted.

The side ear runs a full-session streaming pass in its trained regime. Pause
spans and the native embedding-clock mask are reused from the original
pause-mask-trained-stream implementation: primary blank-labelled ayah or marked
inner-waqf boundaries open a span, and the primary step onto the following word
closes it. Primary listener feeds remain identical when its manifest has no mask
input. The side pass supplies no tracker tokens or positions.

A read-only checker observer records confirmations from every non-graph
wrong-word path, including lexical word-local candidates. A lexical accusation
names its rival; a non-lexical accusation names its existing support bar. No new
margin or confidence cut is introduced. The same checker judges the side stream
using the primary tracker's position events. Agreement requires the same word
and last word, overlapping acoustic visits, and the same rival or identical bar.
The certificate retains the side slot's tokens and posterior rows together with
the rejudged accusations.

Certificates ride the first primary phoneme, following the folded teacher-insert
transport route through evidence JSON and native-frame rechunking. The checker
can only downgrade an otherwise confirmed non-graph wrong word to uncertain.
Certificates match the primary accusation's exact visit and tracker acquisition
clock. Core stored-evidence replay and the live SDK's one-second acquisition
clock receive separate certificates.

The internal primary confirmation history is retained even when the public
warning is vetoed. This is necessary because the folded swap-tail rule consults
that history when assigning surplus speech; changing it could create a new
addition warning. The live SDK reports any side-ear retraction at end(), and
filters that warning from mistakes(). The implementation therefore requires a
full-session side pass and is not an immediate streaming veto.

Primary teacher certificates are completed before the side pass and are never
replaced or recomputed on side evidence. Assertions on every fresh decode reject primary manifest/evidence mutation and
any byte change to primary tokens, posterior rows or folded teacher certificates.
Paired checker replays on both acquisition clocks also reject any changed
non-wrong-word decision (including teacher wins) or newly created warning before
the evidence can be saved for scoring.
The registry loader tests enforce preservation of the primary model object and
manifest, with release and asset-integrity checks.

## Validation

Modal run 1 inspected only the authorized model mount and this experiment's
scratch. `/lab/data/models` contains the shipped model and older channel,
continuation, paired, sparse and targeted variants, but no pause-mask export.
This experiment's scratch contains only its diagnostic runs. The exact final
checkpoint cannot currently be loaded, so none of the four target slots has
been acoustically tested. The user has been asked to expose the saved export
through an authorized mount; no alternate storage route is used.

Modal run 2 built an earlier, in-flight source snapshot and reported two
TypeScript integration errors: the waqf helper was not exported and the new
certificate lacked its acquisition-clock value. Both were corrected before
run 3, which requests the supplied six-shard build/test harness.

Unit coverage includes unchanged default lexical confirmation, exact-rival and
support-bar agreement, stored disagreement veto, visit and acquisition-clock
isolation, same-ear and correct-reading side rejudgment, JSON and rechunking
transport, unchanged primary tracker events, preserved teacher certificates,
mask geometry, asset-integrity failures and primary/side resource release.
The primary invariant also rejects modified rows and teacher certificates.
Run 3 passed both builds and all existing tests in the six-shard harness, but
four newly added core agreement assertions used a held parallel-name swap that
did not reach the non-graph observer. The corresponding SDK tests passed. Run 4
repeated those core failures on the earlier fixture. The core agreement fixture
now uses the existing unrelated lexical-swap example, and held accusations retain
their frame descriptor before window trimming. Run 5 requests the full harness
with that correction, end-of-session isolation tests and decode-time gates. Run 5 passed both builds and all six shards: 760 core tests and 36 SDK tests
passed, with one optional SDK integration test skipped. The harness accepted
its documented Vitest RPC timeout after all tests in that shard passed. Runs 6–7 exposed an incorrect extra test assumption: an attested parallel
phrase can be confirmed by the lexical path, so that fixture does not establish
a graph-only warning. The invalid new assertion was removed; existing parallel
phrase tests are unchanged. Run 8 requests the full harness on the final code,
including rival identity normalisation across string and side-graph paths.
These are unit checks, not acoustic target results.

Run 8 passed both builds and every shard (all six status codes are 0): 760 core
tests and 36 SDK tests passed; one optional SDK integration test was skipped. The
harness accepted only its documented RPC deadline after all tests passed.
`git diff --check` passed. The engine and tests are committed as `e537901d` and
`5c94788d`.

Run 9 recursively inventoried the complete authorized `/lab/data/models` tree,
including nested files and every manifest. There is no pause-mask export or
checkpoint, no manifest with a pause-mask input, and no folded teacher ONNX asset
or external-data file. The shipped primary has its expected SHA256
`714e98addee73b1b8781ad7d99b6d3c745acd9116638baf4e12891f53a15af1b`.
No directories outside the supplied model mount and this experiment’s own scratch
were searched. The exact pause ear and folded teacher files remain required for
a complete bundle; no acoustic diagnostic or target count is claimed.

The generic SDK diagnostic `src/bench/pause-agreement-cli.ts` requires both
side assets. It compares the bundle's primary manifest bytes with a supplied
original directory and checks the primary model hash. For each supplied WAV,
normal SDK decodes with and without the pause ear must have identical primary
evidence, teacher certificates, native-clock tracker events and non-wrong-word
mistakes. A new warning rejects the check. It prints each accusation and its
side-ear agreement, without scoring labels or identifying targets in engine code.
Run this on the supplied phone and studio recordings before publishing a model
selector or requesting practice scoring.

Practice scores requested: 0 of 3. No `.autolab/model.json` is published because
there is no complete verified bundle. No `.autolab/outcome.json` is written:
weights being unavailable is not the user-defined premise kill. The scorer,
lab tooling, test data, supplied manifests and evidence are untouched.

## Limits and remaining work

The unit tests verify mechanics with synthetic evidence, not the proposed
acoustic separation. The earlier global-decode wins do not prove agreement on
these slots while the shipped tracker remains in charge. The exact saved export
and folded teacher files must be exposed, copied into this experiment's scratch
bundle with an unchanged primary model/manifest, and pass the paired diagnostic.
If fewer than three target accusations are vetoed, revert the engine change,
record that premise result and write the requested killed outcome. If at least
three are vetoed, request practice scoring before judging losses. Only the
supervisor's fresh decode and fold gate can establish overall, studio,
clean-session, reader-removal and incident-churn results.
