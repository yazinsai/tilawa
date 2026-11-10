# Context-keyed acoustic crop cache

Status: killed on the source-level premise. Engine code and the experiment card
are unchanged. No practice score was requested.

## Prediction and mechanism

The card predicts one recovered addition, no losses, listener −1 and overall −1
on lab-v1.5u. The proposed failure is a cache hit that supplies warm, verse-biased
acoustic rows to a request for a context-free encode of the same samples.
The proposed key would identify the model artifact, sample interval, feature
contract, attention mask, positional origin and incoming recurrent state.
Candidate discovery, exclusions and teacher verification would remain unchanged.

The experiment stops if none of the targets can have a cross-context acoustic
cache hit whose uncached requested encode restores a missing planted consonant.
The source audit rules out the required cross-context hit before an acoustic
restoration test is needed.

## What the cache actually shares

The sole acoustic-result cache in the SDK is the local `Map` inside
`addInsertionEvidence` (`packages/sdk/src/seam-insertion.ts:54`). It is allocated
after candidate discovery and discarded when that invocation returns. Its only
lookup and insertion are in the sequential teacher-candidate loop. Neither
primary streaming inference nor cold gap inference reads or writes this map.

For a hit on `start:end`, the complete inference semantics are already fixed:

| Requested identity | Current invariant |
| --- | --- |
| Model artifact | One insertion teacher on the invocation's loaded model. The Node adapter lazily opens one fixed teacher backend and retains it. |
| Input samples | One retained audio buffer. The key determines `round(start * frameMs * 16)` through `min(audio.length, round(end * frameMs * 16))`; `frameMs` is fixed for the invocation. |
| Feature contract | Every call uses `teacherFeatures`: fresh Kaldi fbank, per-crop unbiased normalization, stride-two stacking and the same padding. There is no alternate feature mode at this call site. |
| Attention mask | Constructed solely from that crop's feature count, including the unpaired-frame mask. Equal sample intervals produce the same mask. |
| Positional origin | Every teacher decode starts locally at zero. The cached rows are placed at the same keyed absolute `start`; no positional-offset input is passed to inference. |
| Incoming recurrent state | None. Teacher inference receives only `input_features` and `attention_mask`; decoding's CTC state is initialized inside each call. |
| Posterior representation | Every call requests the same full inventory size plus blank, using the same invocation's reference index. |

`createInsertionTeacher` implements those inputs in
`packages/sdk/src/insertion-teacher.ts:26`. Its frontend creates new PCM,
features and masks for each request. The model loader's retained promise caches
the opened backend, not acoustic results or recurrent state
(`packages/sdk/src/node.ts:37`). Different lexical hypotheses sharing a crop
still compute their own sequence marginals after the cache lookup
(`packages/sdk/src/seam-insertion.ts:85`).

The name “warm crop” describes the four seconds of real left audio included in
the selected interval. It does not mean an incoming warm encoder state. Changing
that audio context changes the keyed start or end. Keeping the same complete
interval keeps the same acoustic context.

Cold gap decoding instead creates a new listener for each crop
(`packages/sdk/src/gap-decode.ts:50`). That listener creates its own frontend and
zero-initialized streaming states (`packages/sdk/src/listener.ts:38`). It does
not consult the teacher crop map. Session completion awaits these gap encodes
before invoking insertion verification (`packages/sdk/src/session.ts:179`).
Thus a cold seam request cannot receive a warm cache entry through the proposed
route, irrespective of which target session reaches it. Adding constant model,
feature, mask, origin and state identities to the existing key would distinguish
no currently conflated requests.

## Validation and limits

Read `AGENTS.md` and audited the SDK cache definition, all of its accesses, both
crop callers, teacher frontend, streaming state initialization, backend adapter
and model loading. A local source assertion check confirmed the single local
crop cache, fixed teacher arguments and independent uncached gap listeners.
This was a structural check, not a unit test or acoustic replay.

Two jobs were submitted through the supplied supervisor:

- Modal run 1: `bash tools/autolab_tests.sh`, six shards, for the existing build,
  core and SDK test suites.
- Modal run 2: read-only inspection of the supplied practice data layout to
  prepare the target request census.

Both jobs failed with `NotFoundError: No such file or directory`, with reported
compute cost zero. Neither returned container logs or test results. Consequently
no target audio or evidence was inspected, no dynamic cache-hit census or
uncached teacher comparison ran, and no tests are claimed to have passed.
The finding is limited to the inspected product implementation and its normal
model adapter; it does not measure recognition quality or arbitrary custom
stateful teacher implementations.

The five registered targets were not acoustically evaluated. Target recoveries
are zero because no engine change remains, not a measured five-case acoustic
result. No practice delta is reported. The kill decision follows the absent
cross-context cache route, independently of the infrastructure failure.

Temporary diagnostic instrumentation was removed. No engine behavior changed,
so no new behavior tests were retained. No scorer, lab tooling, test data,
manifest, evidence or experiment-card edits were made. No new model assets were
declared and no fold was performed. The killed outcome is recorded in the
untracked `.autolab/outcome.json`.
