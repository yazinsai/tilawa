# Open edges in pause-bounded utterance discovery

Status: implemented; target validation and practice scoring pending because the supplied Modal supervisor failed before execution.

## Preregistered prediction

Experiment `listener-open-edge-utterance-parse`, listener, explore. Practice set `lab-v1.5u`, card listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. Predict one recovered added word, no losses: listener −1 wrong call and overall −1. That prediction alone is below the fold gate's three-call minimum. The user's build-stop rule additionally requires at least three target wins before a practice score.

## Mechanism

The existing listener discovery checker tests complete pause-bounded utterances at ayah and marked inner seams. The new parser also enumerates a nonempty middle after assigning an optional proper suffix of the preceding pronunciation and an optional proper prefix of the following pronunciation to the utterance's edges. Each assignment consumes separate observed tokens. Boundary matching collapses held sounds and permits only the existing soft substitutions; it introduces no inserted or deleted boundary phones. Cuts inside held madd or ghunna runs, including long noon and meem runs, are excluded.

The original whole-utterance candidate remains first. Additional splits are ordered by middle length and all eligible splits in that utterance reach the discovery callback, even after an earlier split is permitted. Every middle still passes the existing complete lexical spelling, repeat/prefix, local-substring, non-Quran phrase and posterior neighbour-excision checks. The SDK submits those candidates to the existing original-audio warm teacher and unchanged insertion marginal test. Certificate replay enumerates the same splits and requires the same spelling/frame certificate with marginal strictly above 0.5, keeping the first permitted candidate.

The parser is enabled in SDK discovery and certificate replay. Primary-only checking does not enable it. Utterance boundaries, acoustic inference, primary phones/posteriors, tracker input, teacher crops and verification thresholds are unchanged. No asset or model declaration is needed.

## Validation

Supplemental local checks ran copies of the core sources under Node 22's experimental TypeScript transformation, changing import extensions only and adapting the two Vitest files to `node:test` with strict assertions. All 15 cases passed:

- Six parser cases cover optional edges and all splits, mismatched/wrong edges, existing soft versus hard substitutions, disjoint consumption and a nonempty middle, held-run boundaries, and preserving whole-neighbour words for the existing path.
- Nine seam-excision cases include the seven existing controls and two new integrations. A synthetic utterance containing a matched previous-word tail, a complete lexical middle and a matched next-word prefix is accepted only with parsing; a mismatched tail is rejected. The parsed middle replays through ordinary certificates, while missing, misaligned and marginal-at-or-below-one-half certificates stay rejected.

The new integration fixture retains the following-word visit rather than blurring its onset. The longer fragment-bearing utterance plus the original fixture's deliberate onset blur removed that visit, so discovery never ran. Site selection was not widened to compensate. The existing blurred-onset whole-word case continues to pass.

The SDK teacher-judgment test now also asserts that discovery receives this parser and enrichment preserves primary phones and posterior rows. Its Vitest execution remains pending.

The required six-shard `bash tools/autolab_tests.sh` request failed twice (supervisor runs 2 and 4), each before execution with `NotFoundError: No such file or directory`. The initial environment listing (run 1) and supplied p07 trace (run 3) failed with the same error. All four report zero compute spend. There are no successful Modal logs, target traces or fresh target decodes from these requests. No practice-score requests were made.

## Limits and disposition

The five targets remain unmeasured: p07 `37:36:1`, p31 `4:170:12`, p33 `26:208:1`, p57 `51:18:1`, and p79 `20:40:25`. The synthetic tests establish the parsing and certificate behavior, not that any target has a recoverable middle or passes the original-audio teacher gate. Target wins, overall change, clean-session false warnings and studio change are unknown. The complete core/SDK suites and TypeScript builds are also unverified because the worktree has no installed dependencies and the authorized execution path failed.

The implementation is left in place for supervisor validation. Infrastructure failure does not establish the acoustic kill premise or fewer than three target wins, so no killed outcome is claimed. If a successful target audit finds no eligible recoverable middle or fewer than three wins, the engine/test changes must be reverted and the killed result recorded. If at least three targets are won, practice scoring must precede a gate decision.
