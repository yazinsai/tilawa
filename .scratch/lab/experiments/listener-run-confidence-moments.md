# Primary acoustic run confidence moments

Status: killed — the proposed mean-aggregation defect is absent.

## Prediction and proposed mechanism

The card predicted listener −1 and overall −1 on lab-v1.5u: one fewer wrong-word false warning, with no predicted losses. It proposed retaining each primary run's probability sum, contributing-frame count and original frame interval, merging those moments across chunks, then computing frame-weighted mean confidence. Token identities, timestamps, posterior rows, contributing-frame semantics and confirmation bars were to remain unchanged; only an incorrectly aggregated confidence could change a decision.

## Source audit

The implementation does not compute a run mean:

- `packages/sdk/src/listener.ts` initializes a token with the winning probability rounded to four decimals. Each further winning frame in that emitted interval updates `conf` with `Math.max(open.conf, round4(p))`. It never averages chunk summaries, nor divides a probability sum by a frame count.
- A run at an inference window's end is emitted once, with its end at that window boundary. Continuation frames in later windows are suppressed by `carried`. The existing SDK test explicitly requires this early emission and single-token behavior. These frames do not extend the emitted interval or contribute to its confidence. Including them in a later mean would change the contributing-frame definition rather than repair its aggregation.
- `modal/student_decode_cpu.py` also uses a maximum over winning frames; unlike the SDK's early-emission path, it extends saved run intervals as the run continues. `packages/sdk/src/insertion-teacher.ts` independently uses the same peak statistic.
- `packages/sdk/src/session.ts` transports token objects through the evidence grid without combining confidence summaries. The gap decoder preserves token objects when flattening chunks and shifting crop coordinates.
- `packages/core/src/checker.ts` computes `clarity()` as an equal-weight mean of token confidences in the issue span, and `changedClarity()` as an equal-weight mean of the changed tokens' confidences. The word “mean” in these functions refers to averaging tokens, each carrying a peak, rather than averaging contributing acoustic frames within a run.

Maximum aggregation is associative and independent of frame counts. Replacing correctly computed peaks with frame means changes the evidence statistic even on a run entirely inside one chunk. That violates this experiment's requirement that correctly aggregated runs remain identical. No confidence correction within the stipulated mechanism is available, so no engine change was made.

## Validation

Source inspection establishes the premise rejection independently of runtime diagnostics. It covers the actual SDK run accumulator, cross-window continuation, token transport, both checker confidence consumers, and the saved-decode producer. There are zero eligible arithmetic fixes under the card's preserved-statistic requirement; targets won: 0.

The four target posterior audits and an unequal-confidence synthetic SDK run were requested through the supervisor on four shards. Both audit requests returned `NotFoundError: No such file or directory` before any command output or audit artifact was returned. Consequently, runtime targets tested: 0, synthetic runs completed: 0; no numerical target confidence or counterfactual verdict is claimed.

The required `bash tools/autolab_tests.sh` build and core/SDK test run was requested on six shards. Both the initial request and its retry returned `NotFoundError: No such file or directory` before producing test results. No builds or tests completed. No unit tests were changed because no engine behavior was changed. This kill is based on the source-level premise contradiction, not on infrastructure failure or collateral losses.

## Limits and disposition

This result rejects a mean-aggregation accounting bug; it does not establish that peak confidence is well calibrated or that a frame mean could never help a separately specified experiment. The early-emission interval is an explicit existing behavior, and this experiment does not change its timing or evidence protocol.

No practice score was requested. No model, asset, manifest, test data, scorer or lab tooling was changed. There were no engine changes to revert. The experiment card remains untouched; the killed outcome is recorded in untracked `.autolab/outcome.json`.
