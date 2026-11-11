# CTC lattice composed with a normalized phone-confusion channel

Status: implementation prepared; blocked on the supervisor's Modal executor.
The training table is not fitted or enabled, and the experiment is not ready for
practice scoring. The experiment card is unchanged.

## Prediction and mechanism

The card predicts two false substitutions removed and one caught mistake lost:
listener −1 and overall −1. Its four registered targets are the phone warnings at
26:106:1 (p11), 55:48:1 (p36), 26:174:8 (p55), and 92:13:1 (sob-049).
The task's final instructions require at least three target wins before a practice
score; fewer than three measured wins would kill the experiment. No target count
has been measured.

`packages/sdk/src/composed-channel.ts` computes
`sum_y P_channel(y | written) P_CTC(y | acoustic rows)` exactly. It marginalizes
both edit scripts and raw CTC frame paths. The channel uses the existing
training-count estimator for substitutions, deletions, and insertion rates.
With insertion-rate sum λ, each boundary has insertion-loop exit probability
`1 / (1 + λ)` and insertion probabilities `rate[h] / (1 + λ)`.
Substitution and deletion arcs also pay the exit probability. The terminal
boundary exits explicitly. This makes the emitted-string distribution proper,
preserves the insertion loop's expected count, and charges every deletion.
Acceptance consumes every written phone; an unfinished prefix is not accepted.

The acoustic DP tracks written-phone position and the previous raw CTC class.
Blank frames and same-class continuations produce no new channel emission.
Deletion closure runs only at a fresh emitted label or final acceptance, avoiding
multiple counting of identical edit scripts on blank/held frames. Runtime is
O(frames × written length × inventory), with O(written length × inventory)
working memory. It uses the existing posterior-row reconstruction and clock.

The matched control sums edit scripts for the greedy collapsed string under
the **same** normalized channel. A certificate requires four finite scores,
strictly higher joint evidence for the written pronunciation, and strictly
higher fixed-string channel evidence for the existing accused pronunciation.
There is no additional fitted margin, candidate enumeration, or path pruning.
The scorer receives the checker's complete existing hypotheses and original
interval, including the same neighbouring words where its comparison uses them.

The checker consults certificates only at final supported string-path
confirmation, including frozen-slot decisions. It retains its baseline internal
decision for ownership and swap-tail absorption, while downgrading the published
accusation to uncertain. Thus the veto cannot change subsequent insertion
decisions through the internal confirmed-event ledger. Graph confirmations,
primary tokens, and tracker input are unchanged. Certificates are keyed to the
word, both complete hypotheses, and the exact frame interval, and survive normal
JSON evidence serialization on a primary token.

The SDK accepts an explicit `wrongWordChannel` table for live comparisons and
recorded-evidence enrichment. Its discovery replay observes baseline decisions
without feeding preferences back into the checker. No default table is enabled:
shipping existing unrelated counts or an unfitted table would not test the
specified help-train mechanism.

`packages/sdk/src/train/composed-channel-cli.ts` prepares the fitting path. It
accepts only help-train labels, selects none-labelled training takes, excludes
every speaker with a non-training take and an explicit exclusion list, and uses
the unchanged shipped listener plus `accumulateChannel`. Three SHA256-ordered
speaker groups provide disjoint-speaker channel predictions. Construction and
Laplace smoothing are fixed; no practice parameter selection is implemented.
It outputs a source table and a provenance/validation report. Actual authorized
label contents, exclusions, training coverage, and joint-score transfer still
require verification on Modal. No scripted mistake is relabelled as canonical
channel supervision.

## Validation and limits

Twelve focused behavioural checks passed locally using Node's TypeScript
stripping and an ignored assertion runner, executing the real engine source.
These are not a TypeScript build, Vitest run, or the required complete suites.
The checks cover exhaustive edit-script sums and total emitted-string mass;
exhaustive raw CTC paths with repeats, blanks, and deletions; reduction to
ordinary complete CTC for an identity channel; epsilon-deletion counting;
unknown phones, missing intervals, and inventory mismatch; and a synthetic
strict reversal between joint and greedy-string scores. Checker checks cover
exact serialized certificate matching, primary tracker equality, event equality
except the vetoed confirmation state, a session containing both a swap and an
addition, and a verified parallel-graph confirmation that never queries the
scorer. `git diff --check` also passed.

Supervisor Modal runs 1, 2, 3, and 4 all failed before command execution with
`NotFoundError: No such file or directory`. Run 1 requested help-train/data
inspection, run 2 a minimal executor check, run 3 the required six-shard
`bash tools/autolab_tests.sh`, and run 4 a renewed help-train availability check.
There are no command logs, data-derived training results, or target comparisons
from these runs. Their request/result files remain ignored under `.autolab`.
The user was asked to restore the supervisor executor; no repair was confirmed.

No practice scores were requested (all three remain available). No model bundle,
auxiliary asset, or `.autolab/model.json` is declared. There is no frozen fitted
channel in the SDK, so decoding the default model now would exercise no composed
veto and must not be reported as a measured zero gain. The registered target
premise, caught-swap losses, clean-session safety, studio behavior, and fold-gate
result are unknown. An executor outage is not a scientific kill; no killed
outcome or fold is claimed.

To finish, restore the authorized Modal executor, inspect help-train training
speaker coverage and exclusions, run the fitting utility, verify and install its
frozen source table, enable it for SDK decoding, and pass the supplied full test
harness. Then compare unchanged primary evidence and tracker events and measure
the four target decisions on their engine-requested intervals. Fewer than three
target wins requires rollback and a measured kill write-up; at least three
requires an authorized practice score before deciding anything further.
