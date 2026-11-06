# Pause-mask checkpoint as a seam-scoped SDK asset

Status: implementation committed; acoustic validation blocked by the unavailable
existing checkpoint/export. The experiment card is unchanged.

## Prediction and mechanism

The card predicts listener −2 and overall −1. Reuse the final 800-update
pause-mask-trained student without training or selecting new weights. Its int8
export is 72,403,248 bytes, SHA256
`22c8e197f2e60955f0b7794a8198db4d5ae9ee57aa679d3fc163075092fd8c58`.
The prerequisite is at least three registered target additions confirmed by the
unchanged spelled-surplus rules or folded teacher insertion gate.

The parent installed its ear through a declaration in the primary manifest and
substituted scoped rows at the shared checker entry. This experiment declares
the auxiliary file in SDK registry code, validates its identity and streaming
contract, and loads a separate backend beside the caller's unchanged primary
manifest. The primary listener implementation and its ONNX feeds are unchanged.

Seam selection reuses the parent's generic detector-labelled pauses and
in-order tracker steps at ayah boundaries and mushaf-marked inner boundaries.
Each span begins at pause onset and ends at the step onto the second audible
word. ONNX calls occur only on the original-grid windows intersecting those
spans; skipped windows advance positions with empty caches. The trained
attention/convolution mask excludes pre-pause history. Only span rows and tokens
are transported, on the first primary token.

Primary teacher discovery and certificates are completed before side evidence
is consulted. The side review computes its own teacher certificates. The core
checker retains an independent primary checker with unchanged issues and event
sequence, and uses a separate checker for the side evidence. Only that review's
confirmed lexical additions contained entirely in a declared span are published.
All other review calls are discarded. Primary certificates and primary mistakes
remain intact, including teacher insert wins. No checker cost or cut changes.

## Verification and limits

The engine diagnostic `packages/sdk/src/bench/pause-isolation-cli.ts` refuses to
run unless both declared side ears are installed. It checks original primary
manifest bytes and the primary ONNX checksum. For every supplied WAV it compares
normal SDK runs with and without the pause ear: serialized primary evidence,
all folded primary teacher certificates, tracker events, and primary mistakes
must match exactly. Only the new side metadata and side additions may differ.
This diagnostic reads WAVs and writes a verification report; it does not score
or change labels, manifests, or supplied evidence.

Unit tests cover mask geometry, skipped encoder windows and preserved positions,
JSON/rechunking transport, invalid clocks, registry loading and integrity,
release, and preservation of a primary wrong-word accusation when the alternate
review blanks that speech. All existing tests retain their enforcement limits.

Modal run 1 requests read-only checkpoint/model/input discovery. Run 2 requests
the supplied six-shard build and unit-test harness. Neither is a practice score.
Results and outstanding validation are recorded below. No target or population
acoustic conclusion is claimed.

Runs 1, 3 and 5 found no pause-mask export in the supplied read-only
`/lab/data/models` directory. Run 6 checked the training image's supplied
`/student/prior-runs` directory; no pause-mask run or checkpoint is present there
either. The current experiment's persistent scratch contains only its source
snapshots and diagnostic runs. No alternate checkpoint was substituted, and no
training was requested ($0 GPU spend).

Run 4 passed the focused core transport/isolation tests and the SDK mask/session
tests. Its loader test failed because the release method had been wrapped and
the assertion no longer referenced the original spy. The test now captures
that spy before loading. Run 7 requests the complete six-shard build/test
harness with that fix and all new tests.

Run 7 completed successfully on every shard (each status has exit code 0):
759 core tests and 38 SDK tests passed, with one optional integration test
skipped. Both packages built successfully. The new core preservation test
completed in 1.64 seconds. `git diff --check` also passed. Engine implementation
and unit tests are committed as `85277c7e`; these checks verify the code and
synthetic contracts, not the absent trained model's acoustics.

Acoustic target confirmation, sparse-versus-full masked inference parity and
the complete practice/studio primary-output check require the exact existing
export. The supervisor has been asked to expose it through the allowed mounts.
These are pending validations, not zero measured target wins and not a premise
kill. No practice score, model selector or killed outcome has been written.

Once the export is available, a new scratch bundle must copy the original
primary model and manifest byte-for-byte, install the declared pause-mask files
beside them, and include the folded teacher. Run the SDK isolation diagnostic
on every supplied phone and studio WAV before publishing `.autolab/model.json`
or requesting practice scoring. The diagnostic exits nonzero on changed
primary evidence, tracker positions, primary mistakes or teacher certificates.

No practice score has been requested (0 of 3). No overall/component change or
target-win count has been measured. There is no `.autolab/model.json` because
the required bundle does not exist, and no `.autolab/outcome.json` because
unavailable weights do not establish the user-defined kill premise. No changes
were made to the card, primary listener, tracker, scorer, lab tools, supplied
manifests, labels or stored evidence. The pending dependency is the final
800-update pause-mask checkpoint/export identified above, exposed through an
allowed read-only mount, plus the folded teacher files for its bundle.
