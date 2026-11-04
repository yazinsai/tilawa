# Shipped and wide posterior product

Status: implementation prepared; target replay is blocked by the missing wide-model export.

## Prediction

Listener −2 wrong calls, overall −1 on lab-v1.5 practice. The card predicts two of the five additions previously won by the wide student, with about one loss. Targets are p31 `4:157:24` at 55.47 s, p50 `37:99:1` at 146.47 s, p55 `26:193:1` at 269.44 s, p56 `32:21:1` at 106.76 s, and p79 `20:40:25` at 136.10 s.

The final task instruction requires at least three target wins before a practice score. Fewer than three measured wins would kill this experiment; the card's earlier two-target line does not override that instruction.

## Mechanism and implementation

`posteriorProductModel` in the SDK wraps two loaded streaming CTC backends. Both receive the same fbank tensor on every call. Each retains its own cache tensors, namespaced in the listener's state, including separate state for each session and cold crop. The pooled output goes through the existing listener's greedy CTC, confidence calculation and top-k posterior recording. The existing gap path also consumes the combined model when it re-encodes a crop. Tracker, checker and their constants are unchanged.

The fixed formula is `q(c) = sqrt(p_shipped(c) * p_wide(c)) / sum_j sqrt(p_shipped(j) * p_wide(j))`, implemented as an average of full log-probabilities followed by log-sum-exp normalization. There is no weight option or fitted constant. Pooling precedes top-k truncation. This is the usual equal-weight logarithmic pool; a literal normalized product without the square root would have the same argmax but sharper confidence and support. The distinction was raised with the user before any target measurement.

The helper rejects different token orders, sample/frame clocks, window shifts, fbank settings and output frame shapes. It intersects supported error classes. It is runtime-independent and can be passed as `createEngine({ model: posteriorProductModel(shipped, wide), ... })`.

## Validation

Unit tests cover normalized probabilities, a shared runner-up winning over different individual argmaxes, identical-ear preservation, very small probabilities, incompatible inputs, shared feature tensors, separate cache shapes, push-size invariance and independent concurrent/cold-start listeners.

Modal run 2 built core and SDK and ran `bash tools/autolab_tests.sh` over six shards: 698 core tests and 20 SDK tests passed, with one optional SDK end-to-end test skipped. The core checker shard reported the documented Vitest `onTaskUpdate` RPC timeout after all 98 tests in that shard passed; the supplied wrapper accepted that case and returned exit 0. All six shards returned exit 0. Modal run 3 independently rebuilt both packages and passed the four new posterior-product tests.

Modal discovery run 1 completed successfully. `/lab/data/models` contains the shipped model plus twelve channel variants, and no `wide-clean-b-int8` or other wide model. The available shipped model is `spliced-clean`, ONNX SHA-256 `714e98addee73b1b8781ad7d99b6d3c745acd9116638baf4e12891f53a15af1b`, with 45-frame windows, 32-frame shifts, 40 ms output frames and 98 state tensors. A permitted path to the existing wide export was requested. No other experiment's scratch directory was accessed.

The adjacent diagnostic probe decodes complete phone sessions through the SDK with the combined model, including the folded gap path, then calls the unchanged true-position replay and event matcher. Session IDs and paths are arguments, not engine conditions. It compares against stored baseline evidence without modifying it.

The probe is an unexecuted diagnostic, not a validation result. To run it, embed its contents in the supervisor command as `/app/packages/sdk/product-probe.ts` (the supervisor source archive excludes `.scratch`), build core/SDK, and invoke it with `pnpm --filter @tilawa/sdk exec tsx product-probe.ts <shipped-dir> <wide-dir> /lab/data/lab-v1.5/manifest-v1.json exp-listener-heard-word-must-be-lexical-41570706 <session-ids...>`. It honors `LAB_SHARD`/`LAB_SHARDS` and writes case replay reports to `LAB_OUT`.

## Limits and remaining work

No target replay or practice score has run. Target wins, overall change, clean-session warnings and studio changes are **unmeasured**, not zero. A missing artifact does not disprove the premise, so no killed outcome is claimed.

The equal weights do not mathematically guarantee that a confident shipped frame stays unchanged: a sufficiently opposed wide posterior can outweigh it. That premise needs measurement on real outputs. Compute, memory and on-device latency for two models are also unmeasured.

The in-memory helper is not yet a packaged two-model export. Once the wide artifact is available, run the target probe; if at least three targets win, package the same fixed operation for supervisor decoding, write `.autolab/model.json`, and take the required practice score. If fewer than three win, revert the engine implementation and retain the measured result here.
