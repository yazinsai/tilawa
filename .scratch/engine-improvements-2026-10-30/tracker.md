# Tracker experiments, 30 October 2026

Research note only. No engine, benchmark, evidence or label files were changed. Recommendations below are hypotheses for bounded experiments, not measured improvements or new release gates.

## Executive summary

Start with **local transition commitment and confidence hysteresis**, then **temporal acquisition hypotheses with explicit repeats**, then **selective posterior scoring**. A full Quran-wide HMM rewrite is not the first experiment. There is already a useful local dynamic program; the most concrete defect is how its provisional best path becomes a displayed `locked` position.

Before comparing experiments, reproduce the shipped acquisition cadence. The saved session benchmark feeds 40 ms chunks without `acquireMs`, whereas the SDK sets `acquireMs: 1000`. A small paired replay below proves that this materially changes initial-lock results, sometimes for the worse. It does not establish which cadence is best.

Do not interpret 27/35 cold free-arm misses attributed to the tracker as 27 algorithm defects. The attribution changes context/checker behavior too, and many clips contain speech that genuinely matches another ayah. With neither a known assignment nor preceding/following distinguishing words, intended location is unidentifiable. The older session specification explicitly identified 21/25 tracker-attributed misses as wrong-word/twin clips that never acquired the prompted ayah. Improving that number by forcing the prompt onto the audio would be a misleading success.

## What the current tracker actually does

- Global candidate generation votes with exact phoneme n-grams, skips grams with more than 5,000 occurrences, aligns at most 64 vote bins with weighted edit distance, and filters by PER. Posterior scoring cannot recover candidates absent from that shortlist. See `packages/core/src/tracker.ts:246–302`.
- Global acquisition is a fresh ranking of a bounded decoded history. It locks using weighted fit, unweighted PER, at least ten phonemes/two words, and a cost margin of `max(1, 30/n)` edits. The displayed confidence is a hand-designed product of fit and separation, not a calibrated probability. See `tracker.ts:407–448`.
- `prior` and `priorRange` are capped at half an edit, below the minimum one-edit lock margin. They can reorder a tie but intentionally cannot resolve it alone. Tests pin that behavior in `packages/core/test/states.test.ts:73–120`.
- After one global lock, a DP column spans the current ayah ±2 ayahs. It retains costs for multiple offsets, but one best offset drives recentering; jump arcs originate from that best state. Any word start in the local window costs two edits plus a tiny distance penalty. See `packages/core/src/local.ts:13–24, 74–162`.
- Recent cost averages 16 phonemes. Above 0.3 edits/phoneme, a decoded-only global search runs on a 48-phoneme window. A distant global lock must hold for eight consecutive phonemes. One recovery below 0.3 resets the pending evidence; a global result within two ayahs is left to local tracking. See `tracker.ts:727–756` and `.scratch/bookmark-tracker/issues/09-reacquisition-and-jumps.md`.
- A new local run gets only a three-phoneme cost check before display. Local events always say `locked`, even with zero confidence and PER/recent cost beyond global fit limits. See `tracker.ts:602–630`.
- The initial posterior path exists, but the shipped SDK and session benchmark strip posteriors from tracker input. The tracker itself drops them after its first lock. See `packages/sdk/src/session.ts:94–113`, `packages/core/src/bench/sessions.ts:367–374`, `tracker.ts:181–186, 814–818`.
- Memorization can freeze the expected bookmark when a distant jump/loss commits; that frozen value never thaws through an explicit passage-change operation because none exists. Nearby local skips/restarts generally move expected with observed. See `tracker.ts:579–597, 633–639, 745–747` and the issue 09 notes. Preserve the distinction when adding transitions rather than making location inference silently decide intent.

Existing tests cover clean synthetic forward progress, repeats, restarts, skips, pause invariance, two sustained distant jumps, and a single wrong word. Those passing cases do not establish robustness under noisy repeated openings or marginal local runs. See `packages/core/test/local-tracking.test.ts`, `reacquisition.test.ts`, `phrases.test.ts`, `posteriors.test.ts`.

## Existing evidence and two concrete traces

The checked-in historical scoreboard's full student evaluation reports 189/225 clean clips correctly acquired within five seconds (84.0%), 303/388 overall (78.1%), and no counted false jumps in those short clips. This is separate from the latest partial scoreboard row whose comparable fields are pending. The untracked `docs/benchmarks/session-qud-int8-expanded-v1.json` reports 11 supported-style recordings, nine speakers, 195.69 clean minutes: 93.246% located time, 86 wrong-position events, 6/11 first correct locks within five seconds, 1.763 clean false warnings/5 min. It excludes four hadr recordings and fails `prayer-baqarah-yasin` on evidence mismatch. It is diagnostic professional audio, not learner validation.

Of 8,440 unlocated 100 ms samples, Luhaidan Fatir contributes 4,724 (56.0%); Abdulbasit Raad 990 (11.7%); Luhaidan Haqqah 727 (8.6%). Fixing one recording can dominate the aggregate, so also report paired per-recording changes and leave-one-speaker-out summaries.

Read-only replay of saved student evidence and current compiled tracker produced:

1. **Luhaidan Fatir, actual local false jump.** At audio time 3.96 s the tracker correctly locks `35:1:3`. At 5.64 s it displays `35:3:16`, run start `35:3:15`, while truth is still `35:1:4–5`. At 9.88 s it still displays `35:3:18` with confidence 0, recent cost 0.40 and PER 0.379; at 11.04 s it displays `35:3:19`, confidence 0, cost 0.4625, PER 0.433. It returns to `35:1:7` at 11.92 s. This establishes a provisional local jump and poor-fit lock reaching the UI; it does not by itself establish whether the initiating phoneme error is in the listener or reference.
2. **Minshawi Luqman, repeated ambiguous opening.** At audio time 7.60 s the truth is still repeated `31:1`. The tracker locks a span `2:1–2:2:1`. Its shortlist already contains `31:1–31:2:1` third, so increasing retrieval recall alone cannot fix this example. First correct lock is 22.28 s absolute, 22.22 s after onset. Repeated `Alif Lam Mim` plus noisy continuation was treated as linear progress into a different surah. The repeat detector exists, but is bounded to 8–40 phonemes and operates by collapsing approximate repeated decoded blocks; this case survives it.

These are text/evidence/trace diagnoses, not independent audio judgments. Existing `.scratch/help-review/automated-triage.json` explicitly marks teacher agreement as correlated and retains unresolved audio review; do not promote these observations into label corrections.

## Prerequisite: make the acquisition policy explicit in comparisons

The SDK's `ACQUIRE_MS = 1000` is justified in code by short prefixes causing wrong initial locks (`packages/sdk/src/session.ts:163–164`). `runSession` omits the option and therefore searches on each 40 ms evidence chunk. This means its initial-acquisition result is not the shipped SDK policy.

A paired **tracker-only** prefix replay used the exact saved student evidence, the current compiled code, free mode, the production phrase asset, no posteriors, and 40 ms evidence chunks in both arms. Only `acquireMs` changed. Each recording was truncated at its first truth-word onset +35 seconds; reported delay subtracts that onset. A missing value means no correct locked event within that prefix. The baseline reproduces all 11 stored initial-lock delays exactly.

| Recording | Per-push delay | 1,000 ms delay |
| --- | ---: | ---: |
| Abdulbasit Raad | 16.260 s | 17.140 s |
| Ayyub Juz Amma | 1.900 s | 2.220 s |
| Husary Fath | 4.600 s | 4.960 s |
| Jaber Inshiqaq | 3.620 s | 4.060 s |
| Luhaidan Fatir | 3.580 s | 3.580 s |
| Luhaidan Haqqah | 5.540 s | no lock in prefix |
| Luhaidan Qiyamah | 10.280 s | 10.560 s |
| Mandour Taha | 7.220 s | 9.620 s |
| Minshawi Luqman | 22.220 s | 23.700 s |
| Taraweeh Maryam | 2.140 s | 2.220 s |
| Witr Muawwidhat | 1.885 s | 8.485 s |

Minshawi's wrong locked events inside these prefixes increase 2→10. This is neither a full-session comparison nor full SDK execution, and does not measure checker effects, audio latency or claim that cadence alignment improves tracking. It is a reason to share the real policy before tuning it. Keep both old and shipped-policy baselines so that a reporting correction cannot masquerade as a model improvement.

## Ranked experiments

### 1. Commit local transitions using sustained evidence; distinguish tentative tracking from locked display

**Mechanism.** Keep the current local DP, but expose the best *distinct run* and its runner-up score. Treat a new repeat/skip/run as provisional; commit after accumulated margin on newly observed phonemes rather than three cheap phonemes. Use separate enter/exit thresholds and evidence decay, so one good phoneme does not erase an otherwise sustained deviation. Preserve the last trusted location while uncertain, with state/uncertainty visible. Do not emit a new `locked` event with zero confidence. Keep the former run alive briefly instead of immediately pruning around a speculative new center.

**Targets.** The Fatir jump above; momentary wrong highlights before weakness activates; repeated weak/healthy oscillation; overconfident local paths after a poor acoustic stretch. This is a direct continuation of the known local-flash limitation in issue 09.

**Minimal seam.** Extend `LocalSearch.position()` with distinct-origin alternatives or a best-vs-runner-up margin, and change the `localEvent`/weak/pending gate. Start with a shadow commit decision over unchanged DP scores. Avoid retuning `subCost` or the checker in the same arm.

**Risk.** Fewer wrong positions can merely become more unlocated time; over-sticky continuity can hide real skips or delay repeats. Memorization must preserve the expected bookmark independently of a newly observed location. A frozen display must not be counted as confidently correct by an altered metric.

**Validation/stop rule.** First reproduce Fatir 0–25 s and existing repeat/skip/distant-jump fixtures, then all 11 paired supported sessions and the clean-help set. Report wrong *seconds* and events, located share, recovery latency, repeat/restart latency and highlight delay. Discard a setting that only replaces wrong locks with abstention without increasing correctly located time, or loses an existing genuine skip/restart. As a prototype criterion, cap added median verified transition/highlight delay at 200 ms; this is an experiment budget, not a proposed release gate. Inspect per-speaker changes before accepting aggregate gains.

### 2. Carry a small set of temporal location hypotheses, including repeat/restart paths

**Mechanism.** Keep, for example, four to eight distinct location/run hypotheses across acquisition cells and reacquisition. Advance each with forward, repeat/restart and outlier transitions. Accumulate new evidence once, then prune; do not add full overlapping-window costs repeatedly. Publish a lock only when an entire path, rather than one favorable window, separates. For an acoustically identical twin, retain ambiguity until a distinctive continuation or real context resolves it.

**Targets.** Minshawi's repeated opening; cold starts at common phrases; twins that diverge a word later; a wrong global lock whose runner-up would recover once another word arrives. Also test controlled 43:3→12:2-style substitution with *real preceding/following context*, not only a cold isolated error clip.

**Minimal seam.** Wrap existing `locate`/`assess` proposals with a short temporal hypothesis manager; retain the current retrieval and local search first. Track canonical reference offset and run origin separately from observed-time occurrence, so the same words repeated twice are two events. Reuse `createLocalSearch` as one hypothesis transition primitive if useful; it already retains local offset alternatives.

**Risk.** Cost scales with active hypotheses; continuity can overpower a genuine change; pruning can still eliminate the correct path. The deterministic Quran order is a useful transition model, not a license to overwrite an actual wrong-word utterance. No temporal model can infer the intended ayah from indistinguishable audio without context.

**Validation/stop rule.** Run Minshawi, all eight `Alif Lam Mim` starts, held-out twin continuations, existing phrase/repeat/jump fixtures and long supported sessions. Report true-hypothesis survival, wrong-lock dwell, first *identifiable* correct lock, exact repeated-occurrence tracking and bounded CPU/memory. Stop increasing beam size once survival gains saturate; reject a design that cannot beat the current false-lock/latency tradeoff on held-out speakers or exceeds the existing tracker runtime gate. Do not count correct abstention before disambiguation as an acquisition failure to be optimized away; report it separately while retaining existing published metrics.

**Literature basis.** Probabilistic score following has explicitly modeled errors and arbitrary repeats/skips with efficient transitions, rather than assuming only forward progress. [Nakamura, Nakamura & Sagayama (2015/2016)](https://arxiv.org/abs/1512.07748) supplies a close structural analogy. Its musical results do not predict Quran accuracy; the transferable idea is a sequence model with explicit jump/repeat alternatives.

### 3. Rescue ambiguous candidates with selective, streaming posterior scores

**Mechanism.** Retain a bounded posterior ring buffer and CTC-score the top few competing paths at acquisition ambiguity and suspected local divergence. Score each candidate on the same observed frames. Use this as acoustic evidence, with a decoded-only fallback, rather than injecting a Quran language model into the listener's raw evidence. Keep the checker input unchanged and independently scored so a location prior cannot erase a real recitation error.

**Targets.** Greedy substitutions where the correct token is second-best; low-confidence omitted sounds that spuriously favor a local jump; near-twin openings with a truly audible distinguishing consonant. It will not recover information absent from the posterior or distinguish perfect twins.

**Minimal seam.** Reuse `PosteriorFrames` and `posteriorCost`; change SDK/replay wiring behind an experimental option and allow a bounded post-lock buffer. Initially rerank only existing candidates at a fixed event cadence, not every frame, and instrument whether the true candidate was retrieved. Only add alternate-token retrieval if failures prove it is missing.

**Risk.** `NATS_PER_EDIT = 3` is currently hand-set; top-k truncation, held-sound normalization and mixed cost scales can induce false certainty. Full-window rescoring at every phoneme wastes CPU. The historical issue 05 reports mixed posterior results and a known unintelligible false lock, so enabling the existing option is not an established win.

**Validation/stop rule.** Paired decoded-only/selective-posterior runs on the same student evidence and same acquisition policy; split tuning/evaluation by speaker. Begin with Minshawi and noisy Fatir, then clean-help and all supported sessions. Report shortlist recall, ranking accuracy conditional on retrieval, wrong-lock dwell, located time, lock delay and runtime. Reject if calibration must use evaluation speakers, if clean false locks increase, or if gains disappear after matching acquisition cadence. If reranking rarely changes errors because true candidates are absent, stop this arm and investigate retrieval instead.

**Literature basis.** [Graves et al., CTC (2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) explains why the most probable framewise path need not yield the most probable label sequence and gives prefix-probability decoding. This supports retaining alternative acoustic evidence; it is not evidence that a particular Tilawa posterior weight is calibrated.

### 4. Learn how much to trust genuine session context, while keeping observed and expected bookmarks separate

**Mechanism.** Represent user-selected surah/page/assignment/resume and previously verified continuity as provenance-tagged priors. Use a mixture with nonzero outside-context probability, allow strong acoustic evidence to escape, and decay stale resume context. Evaluate contextual ambiguity resolution separately from an acoustic lock. Add an explicit passage-change/reset operation for the expected bookmark instead of its current permanent post-jump freeze. A nearby observed skip must remain distinguishable from the expected progression in memorization mode.

**Targets.** Common opening phrases where the app genuinely knows a surah/page; stale resumes; false expected-position advancement after a local skip; permanent expected freeze after the user intentionally starts a different passage. A half-edit bonus cannot solve a true tie today, by design, which explains why the current page/surah arm often barely moves acquisition.

**Minimal seam.** An experimental context policy around `priorBonus` plus explicit expected-bookmark lifecycle at the session boundary. Use only actual context available to the application; never pass the benchmark's true start as a resume hint. Leave raw observed candidates and acoustic costs visible.

**Risk.** Incorrect context can coerce a wrong location or convert a real mutashabihat error into a clean recitation. It can also improve assignment runs only by suppressing out-of-passage evidence. The checked-in tests deliberately prevent a prior alone from locking, so relaxing that is an explicit product-policy experiment, not a harmless threshold tweak.

**Validation/stop rule.** Paired correct-context, stale-context, wrong-page/surah, no-context, and intentional-passage-change sessions; include genuine outside-context jumps and twin substitutions. Measure contextual and acoustic locks separately, escape time, observed-position accuracy, expected-bookmark stability and end-to-end warnings. Stop if improvements depend on oracle start information or wrong context prevents recovery. Do not ship stronger priors without real error continuations: the current clean professional sessions have no error-positive recall evidence.

## Reproducible prefix replay

Run from `/Users/rock/ai/projects/tilawa` against the existing compiled files. This does not write output files. It is the paired replay used above, not a benchmark replacement; it deliberately uses the scorer's published ±1-ayah/two-second-tolerance predicate.

```sh
node --input-type=module <<'JS'
import fs from 'node:fs';
import {createTracker,loadReferenceIndex} from './packages/core/dist/index.js';
import {chunk} from './packages/core/dist/bench/index.js';
import {qudAudit,ayahOrdinal,onQuran,correctPosition} from './packages/core/dist/bench/sessions.js';
const asset=JSON.parse(fs.readFileSync('packages/core/assets/ref-v2.json','utf8'));
const index=loadReferenceIndex(asset), ay=ayahOrdinal(index);
const phrases=JSON.parse(fs.readFileSync('packages/core/assets/phrases-v1.json','utf8'));
const report=JSON.parse(fs.readFileSync('docs/benchmarks/session-qud-int8-expanded-v1.json','utf8'));
for(const row of report.rows.filter(r=>r.effectiveContext==='none')) {
  const id=row.id;
  const rec=JSON.parse(fs.readFileSync(`.scratch/bench/qud/${id}.json`,'utf8'));
  const {session:s}=qudAudit(rec,index);
  const e=JSON.parse(fs.readFileSync(`.scratch/session-bench/qud/evidence/${id}.evidence.json`,'utf8'));
  e.nFrames=Math.ceil((s.words[0].startS+35)*1000/e.frameMs);
  e.phonemes=e.phonemes.filter(p=>p.start<e.nFrames);
  const cells=chunk(e,.04,false), results=[];
  for(const acquireMs of [undefined,1000]) {
    const t=createTracker(index,{mode:'free',phrases,acquireMs});
    const events=cells.flatMap(c=>t.push(c)); events.push(...t.end());
    const first=events.find(p=>p.type==='position'&&p.state==='locked'
      &&onQuran(s,p.timeMs/1000)&&correctPosition(s,p,p.timeMs/1000,ay));
    const wrong=events.filter(p=>p.type==='position'&&p.state==='locked'
      &&onQuran(s,p.timeMs/1000)&&!correctPosition(s,p,p.timeMs/1000,ay));
    results.push({acquireMs:acquireMs??'perpush',
      lockDelay:first?+(first.timeMs/1000-s.words[0].startS).toFixed(3):null,
      wrongEvents:wrong.length});
  }
  console.log(JSON.stringify({id,results}));
}
JS
```

For the two traces, create the same tracker with `trace:true`, truncate at onset +25 s, and print the locked events alongside `recentWords(s, p.timeMs/1000)`. All incident times above are absolute audio times. One instrumentation caution: a deferred global trace can compute normalized cost after `startLocal()` clears `observed`; Minshawi's reacquisition trace contains non-finite costs serialized as `null`. Preserve assessment-time values before relying on those trace fields for calibration. The ranking/position observations above use the initial acquisition trace or event values, not those invalid deferred costs.

## Boundaries for interpretation

- Keep the 388 short clips for cold acquisition/regression, but tune session behavior on ordered speech with real preceding and following words. Do not fabricate long learner sessions by joining speakers or reuse independently decoded clip evidence as continuous-listener evidence.
- Current expanded sessions cannot validate missed-error recall, warning delay, or true learner behavior. Four hadr exclusions are a coverage limitation, not evidence that all remaining styles are easy.
- Measure exact word/occurrence alignment in diagnostic traces in addition to published tolerant position metrics. The current ±1-ayah score can call the wrong occurrence of a short repeated ayah correct.
- Share benchmark cadence/context/decoder policy with SDK execution before changing thresholds. Report the historical baseline, the matched baseline and the experimental change separately.
- Any future metric-moving implementation still requires a documented scoreboard row under `AGENTS.md`; this note proposes no changes to truth or gates.
