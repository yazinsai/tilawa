# Pause-mask ear agreement for wrong words

Status: complete verified bundle declared in the untracked `.autolab/model.json`.
The actual trained side ear vetoes all four tested target accusations. All seven
paired phone/studio diagnostics pass. The change remains for the supervisor's
normal fresh decode and fold gate; no overall gain or fold is claimed. The
experiment card is unchanged by this continuation.

## Prediction

Keep the shipped decode and ask the existing pause-mask-trained student to agree
with each string-path wrong-word accusation. Predict three of four target
warnings vetoed, with about one caught swap lost: listener −2, overall −1.
The registered kill condition is fewer than three target accusations vetoed by
the built unchanged-checker rejudgment.

## Mechanism

The SDK registry declares the existing final 800-update int8 export, SHA256
`22c8e197f2e60955f0b7794a8198db4d5ae9ee57aa679d3fc163075092fd8c58`,
72,403,248 bytes. Its loader validates inventory, streaming geometry, state layout,
mask clock, byte length and hash. The primary manifest stays byte-identical;
no auxiliary selector is added to it. No weights are retrained or substituted.

The side ear runs a full-session streaming pass in its trained regime. Pause
spans and the native embedding-clock mask are reused from the original
pause-mask-trained-stream implementation: primary blank-labelled ayah or marked
inner-waqf boundaries open a span, and the primary step onto the second word
after the boundary closes it. Primary listener feeds remain identical when its manifest has no mask
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

## Original validation

The earlier implementation commits passed both builds and all six unit-test
shards in run 8: 760 core tests and 36 SDK tests, one optional integration test
skipped. Tests cover exact-rival and support-bar agreement, visit and acquisition
clock isolation, unchanged default/graph behavior, JSON and native rechunking,
primary tracker preservation, teacher-certificate preservation, mask geometry,
resource release and veto-only session completion.

Runs 1 and 9 inventoried only the then-authorized model mount and this
experiment's scratch; no saved ear or teacher files were exposed there. Thus the
earlier implementation had no acoustic result and no published model descriptor.
The feedback's baseline-only decode/score was invalid and held; it is excluded
from the experiment's measurements.

## Completion after artifact feedback

Collected all existing requests 1–9 before requesting compute. None ran the
saved ear. Run 10 confirms both authorized scratch bundles exist. The stored
lab copy named in the feedback is absent from this run's mount, so the actual
saved scratch export is used.

Run 12 verifies the exact trained ear and preserves the original primary and
teacher from `/scratch/listener-word-ear-ssl-ear-agree-adds/bundle`. It assembles
only those four primary/teacher files and the two existing pause-ear files into
`/scratch/listener-pause-mask-ear-swap-agree/bundle`. Both manifests are copied
byte-identically; no missing manifest is generated. The pause training bundle's
other teacher graph is deliberately excluded because it is not the declared
original graph.

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| Primary manifest | 11,044 | `785f5eb4724bc0ecd7217b48066aa34ba6dd8ba77f144f6f422c2436a1fc442f` |
| Primary ONNX | 67,759,744 | `714e98addee73b1b8781ad7d99b6d3c745acd9116638baf4e12891f53a15af1b` |
| Teacher ONNX | 11,342,825 | `5ee50c2b14d7e8e4808ee4adc33e6114f73481b607291136992a5d45444a1e18` |
| Teacher external weights | 2,422,885,632 | `d3f24efba836219f0c556ad25e3a1046f9caf36b2ed639cd283f795c78da58b4` |
| Pause-ear manifest | 11,023 | `5891eb7f5452da611af275622f5279cff6b07d1c35bf36c7be3706e14f39e1ab` |
| Pause-ear ONNX | 72,403,248 | `22c8e197f2e60955f0b7794a8198db4d5ae9ee57aa679d3fc163075092fd8c58` |

Manifest checks compare schema, architecture, blank class, inventory, tokens,
rate, CTC clock, fbank, window, shift, chunk, left context and every state.
The SDK loader now also rejects mismatched frontend, blank class and architecture;
unit tests cover these rejection paths. Run 11 passes both package builds and
all six test shards: 760 core tests and 39 SDK tests, one optional integration
test skipped. Only the harness's documented post-test RPC deadline is tolerated.

Run 13 completed all seven paired ordinary SDK decodes successfully (every
shard exits 0): phone and studio for p11, p36 and p55, plus the fourth phone
slot from the cluster history, realuser-v1-sob-049 at 92:13:1. Every asset hash
matches the declaration; the primary manifest is byte-identical to the original.
The actual ONNX accepts exactly `x`, all declared states and `pause_floor`, with
native int64 mask shape `[1,16]`. Its posterior output is `[1,8,43]`; every cache
output matches the primary state shape and dtype.

The diagnostic uses complete ordinary SDK decodes with and without the ear.
All seven comparisons preserve primary phonemes/rows, tracker events, folded
teacher certificates and every non-wrong-word judgment byte-identically. No new
warning appears. The runtime also checks veto-only behavior on both the native
practice acquisition clock and the SDK's one-second clock before evidence can
be saved.

| Phone target | Primary slot, native frames | Primary accusation | Side rejudgment | Result |
| --- | --- | --- | --- | --- |
| p11, 26:106:1 | `[2593,2601)` | 20-nat bar | no accusation | vetoed |
| p36, 55:48:1 | `[7044,7062)` | 20-nat bar | no accusation | vetoed |
| p55, 26:174:8 | `[2491,2656)` | 20-nat bar | no accusation | vetoed |
| realuser-v1-sob-049, 92:13:1 | `[895,909)` | 20-nat bar | no accusation | vetoed |

On native stored-evidence replay, these are exactly the four removed warnings.
All other warnings in the tested phone recordings survive, including p55's
planted 26:172:2 swap and its confirmed addition at 26:175:1. Studio warning
counts are unchanged: p11 6 → 6, p36 0 → 0, p55 10 → 10. p11's studio lexical
swap retains the exact rival `صَالِحُن`. These are paired diagnostic results,
not practice-set scores or release-gate results.

The registered fewer-than-three-target kill does not fire: 4 of 4 tested slots
are vetoed, including all three card-listed practice incidents. No narrowing,
constant change, new training or checkpoint selection follows the measurements.
The completed bundle remains at
`/scratch/listener-pause-mask-ear-swap-agree/bundle`, with untracked descriptor
`{"path":"bundle"}`. No killed outcome is written.

The earlier baseline-only score remains excluded. Read-only inspection of the
worktree's `serve_practice()` shows a builder request runs the card's currently
selected evidence rather than performing a fresh model decode. Repeating that
request would repeat the invalid measurement identified in the feedback. As
explicitly requested there, the verified descriptor is left for the normal
supervisor fresh-decode/practice/gate sequence. No stale-evidence score is
requested, no overall result is asserted, and no fold decision is made here.

## Limits

Unit tests verify mechanics with synthetic evidence; the seven-recording
diagnostic establishes the targeted acoustic veto count with the shipped
tracker in charge. It does not measure losses across the full practice set. An
agreement veto is delivered at session end in the live SDK. Only a fresh
supervisor decode and fold gate can establish overall, full-set studio,
clean-session, reader-removal and incident-churn results. No manual fold is
requested or performed.
