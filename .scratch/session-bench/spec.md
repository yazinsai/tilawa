# Session bench: tests that look like real use

Status: ready-for-agent

## Implementation review (12 October 2026)

Reviewed against the current core and SDK. Accepted for incremental implementation; the benchmark stays reported-only. Release gates and recorded-session promotion requirements remain unchanged.

- `CheckerOptions.passage` is available, and SDK `startMemorization` uses it. Resume is a weak tracker prior; surah/page remain unsupported and explicitly marked in benchmark rows. No true-start prior is inferred.
- The SDK feeds decoded evidence to the tracker and posteriors to the checker on a fixed evidence grid. The initial replay follows that separation with 40 ms evidence cells; it is evidence replay, not device latency.
- Display is not an SDK-defined UI field. Initial position measures use locked top-candidate position held between events; acquiring/ambiguous/lost states do not count as correctly located. This display policy must be confirmed against the eventual sample app before promotion.
- All labelled errors remain in recall's denominator, including abstentions. Delays apply only to matched warnings and are censored by misses; counts accompany them. First emission assigns an occurrence, later states retain it, and any confirmed state counts even after resolution.
- Clean-speech minutes count timed words, excluding labelled-error intervals. Position samples additionally allow the two-second last-word tolerance. This prevents silence from diluting false-warning rates.
- Omissions use the end of the first occurrence of the second following ayah, or session end when absent; repeated later occurrences must not extend that horizon.
- Existing help labels lack word times. Coverage audit found no minute-long consecutive same-speaker clean runs and no error clip with both clean adjacent ayahs from its held-out speaker. Audio must not be padded with other speakers. Tlog speaker clusters are not independently verified identities and have not been audited for this format.

First increment: session format/validation, lifecycle/occurrence scorer, context adapter, QUD conversion/replay, same-speaker coverage audit and deterministic continuous-WAV builder for a supplied timed catalogue. Baseline contains three original QUD recordings with fresh continuous student decoding and a teacher comparison; provenance and training-overlap limitations are recorded. Scoreboard integration reports session measures only. Full QUD annotation reconciliation, error-session timing/coverage, session review UI, recorded agreement, and threshold selection remain open.

## Problem

The acceptance bench (`packages/core/src/bench/acceptance.ts`, `docs/benchmarks/acceptance-v3.json`) scores 388 clips, median 8.5 s, one prompted ayah each, starting cold. Its two arms are "app knows nothing" (free) and "app knows the exact ayahs" (declared). Neither matches the app:

- Real sessions are minutes long, cross ayah boundaries, and nearly always have some context (chosen surah, page, assignment, resume point).
- Planted errors sit in short clips that end right after the error, so mutashabihat substitutions can never resolve: the recited text *is* another ayah and nothing follows. 21 of the 25 tracker-attributed free-arm misses are wrong-word clips that never latched onto the prompted ayah.
- Recitation (reading) and memorization (from memory) have different costs: in recitation a false warning is the main harm; in memorization a missed error is. One combined recall/precision number hides both.

Result: free-arm wrong-word recall is 1/29 and moves for reasons the app doesn't care about, while what users will see (warnings per session, position jumps, delay to a warning) is only partly measured.

The scoreboard (`scoreboard-cli.ts`) already adds lead-in/lead-out checks (ayahs before and after the clip, from tlog) and listener/runtime gates on top of these clips. This spec replaces the clip-based *error and position* gates only; listener, runtime and lead-in gates stay as they are until a session measure covers them.

## Goal

A release bench whose items are **sessions with context**, scored per **use track**, so a number moving means users' experience moved.

Non-goals: replacing the listener bench; dropping the existing clips (they stay as unit tests for cold acquisition and alignment).

## Design

### Session item

```ts
interface BenchSession {
  id: string;
  track: "recitation" | "memorization";
  speaker: string;
  source: "stitched" | "recorded" | "qud";
  audio: string;              // one continuous file, or a manifest of same-speaker segments
  context: SessionContext;    // what the app would know at session start
  /** Ordered, timed truth: what was actually recited, occurrence by occurrence. Repeats get their own spans. */
  spans: (
    | { id: string; kind: "recitation" | "repeat" | "restart"; from: WordRef; to: WordRef; startS: number; endS: number }
    | { id: string; kind: "non_quran"; label: string; startS: number; endS: number }  // takbir, isti'adha, talk
  )[];
  /** Per-word times within spans; position scoring needs them. */
  words: { span: string; at: WordRef; startS: number; endS: number }[];
  events: SessionEvent[];     // errors
  nonErrors: { span: string; at: WordRef; startS: number; endS: number; kind: "repeat" | "restart" | "self_correction" }[];
}

interface SessionEvent extends LabelEvent {
  /** Which occurrence of the word (span id), so repeated visits and twin copies are told apart. */
  span: string;
  /** Audio interval of the error; for an omission, the gap where the word belonged. Delay is measured from `endS`. */
  startS: number;
  endS: number;
}

type SessionContext =
  | { kind: "none" }
  | { kind: "surah"; surah: number }
  | { kind: "page"; page: number }
  | { kind: "assignment"; from: WordRef; to: WordRef }
  | { kind: "resume"; at: WordRef };
```

Each session is scored at its own context and, where meaningful, at every *weaker* context too (assignment → surah → none). This replaces the free/declared split with a curve: how much the engine leans on context.

### Sources, cheapest first

1. **Stitched sessions** (this week). Concatenate same-speaker clean clips covering consecutive ayahs into one stream, with the speaker's natural pauses (or 300–900 ms silence). Splice in that speaker's planted-error clips where they belong in sequence, so every error has real recitation before and after it. Same speaker throughout, so no voice change at the seams. Speakers with too few consecutive clips are skipped, not padded with another voice.
2. **QUD long-form** (`.scratch/bench/qud/*`). Long professional recitation with word-timed truth: the recitation track's false-warning and tracking baseline at realistic length. Not all contiguous: some are partial passages, prayer audio (`taraweeh-maryam`, `prayer-baqarah-yasin`) with takbir and Fatiha restarts, repeated words (`mandour-taha`, `minshawi-luqman`) and surah jumps. Converted to ordered `spans` from `truth.words`; isti'adha, basmala and non-Quran audio become `non_quran` spans and are excluded from clean-speech minutes. No planted errors, fluent professionals: easy end only.
3. **Recorded sessions** (later, the ground truth). 30–50 real sessions: memorizers self-testing a passage, readers reading a page. Labelled for what actually happened, including self-corrections. Used to check that stitched numbers predict real ones; if they diverge, stitched sessions get rebuilt, not trusted.

Planted errors in stitched sessions are placed deliberately: ayah start, middle and end; mutashabihat swaps (tagged `forms_other_ayah`) followed by at least one more ayah; skips at ayah boundaries.

### Scoring, per track

All tracks report, separately for each context level:

| Measure | Recitation | Memorization |
|---|---|---|
| False warnings per 5 min on clean speech | **gate** | reported |
| Error recall (wrong/omitted word, skipped passage) | reported | **gate**, mutashabihat slice reported separately |
| Precision of warnings | reported | **gate** |
| Wrong position shown, per session (see below) | **gate** | **gate** |
| Located: first correct lock within 5 s of recitation onset, and share of recitation time correctly locked | **gate** | **gate** |
| Delay from the error's `endS` to its first *suspected* and its *confirmed* warning (p50, p90) | reported | **gate** (confirmed p90) |
| Abstentions: twin ambiguity (tracker) and `uncertain` (checker), each as a share of labelled errors and per clean session | reported | reported |
| Non-errors (repeats, restarts, self-corrections) warned on | **gate** | **gate** |

Thresholds are set from the first baseline run, not guessed now.

**Warning lifecycle.** Checker events are grouped by `id`; each id is one warning, scored once. A warning *counts* (for recall, precision, false warnings and non-error warnings) if it ever reached `confirmed`, as acceptance does today. Suspected-only and `uncertain` ids are abstentions, reported, not counted. A confirmed id later resolved (self-correction) still counts as shown: the user saw it.

**Matching.** One-to-one, by `id`. A warning is assigned to an event by class, word, and time: its *first* emission (`timeMs` of its first state) must fall in `[event.startS, event.endS + W]`, with `W` a fixed 6 s for wrong/added words and up to the end of the second following ayah for omissions (the checker's own pending horizon). Eligibility comes first: a warning is only eligible for an event if no *later* occurrence of the same word has started before the warning's first emission (so once a repeat or twin copy begins, warnings go to it, not the earlier one). Among eligible events, the earliest-ending unassigned one wins. Later states of that id inherit the assignment; they're never matched on their own. So for an identical twin (94:5/94:6) or a repeat, a warning on the wrong occurrence is a false positive, not a hit.

**Abstentions never help recall.** Recall's denominator is every labelled error; an abstention counts as not caught. Abstention rates are reported beside recall, split by cause, so abstaining can't inflate the gate and a twin "not sure" is visible as such.

**Wrong position.** Scored against word-timed truth (every session carries per-word times: QUD's `truth.words`, the stitched clips' alignments, and recorded sessions labelled to the word). At each position event, the shown word must be within one ayah of the word being recited at that moment, or of the last word recited within the previous 2 s (the reciter has just moved). Positions shown during `non_quran` spans or silence longer than 2 s are ignored, neither right nor wrong.

**Located time.** The shown position is a step function: each position event holds until the next one. Sample it every 100 ms over Quran spans only (excluding `non_quran` spans and silence over 2 s). A sample counts as correctly located if the state is `locked` and it passes the wrong-position rule above. Share located = located samples ÷ samples. Locks during `repeat`/`restart` spans may sit on the repeated words. Anything else is a wrong position. Stricter than acceptance's ±2 ayahs (`LOCAL_AYAHS`), on purpose: inside a long session, two ayahs ahead is visibly wrong.

### Engine interface

The engine has no context input apart from the checker's declared passage and the tracker's `prior`. `prior` is one ayah, with a bonus for it and its immediate neighbours, capped at half an edit (`priorBonus`, `MAX_PRIOR_BONUS`); it can't express "somewhere in this surah" or "somewhere on this page". The bench passes context through a single adapter, using only what the app would know, never the session's true start:

| Context | Today | Needs engine work |
|---|---|---|
| `assignment` | checker `passage` (today's declared arm) | none |
| `resume` | tracker `prior` at the resume ayah, weight 1 | none, but weak (half an edit) |
| `surah` | unsupported: scored as `none` and marked so | a range prior in the tracker |
| `page` | unsupported: scored as `none` and marked so | a range prior, plus a page→ayah map (Madani mushaf page table, added to the reference assets) |
| `none` | nothing | none |

Weaker contexts are derived by dropping information: `assignment` → `surah` of its first word → `none`; `resume` → `surah` → `none`. Unsupported rows are still run, so the gap they leave is visible on the scoreboard and closing it is measurable.

This lets the bench land before the latching work, and shows that work's effect as the gap between context levels closes.

### Scoreboard

`scoreboard-cli.ts` gains a session section: one row per (track × context), with gates marked. The clip-dump/review page gets session items. A session is long, so the review page opens it at the event, with a few seconds of audio either side, not from the beginning.

## Plan

1. **Session format + loader.** `BenchSession` type, manifest format, validation (same speaker per session, spans ordered and non-overlapping, every event and non-error inside its span).
2. **Stitcher.** Build stitched sessions from the existing labelled clips and clean clips; deterministic from a seed; output a manifest plus per-session evidence (run the listener on the stitched audio, don't concatenate per-clip evidence: frame boundaries and posteriors must be real).
3. **Session scorer.** Run tracker + checker over a session at each context level; produce the measures above. Reuse `scoreClip` matching where possible.
4. **QUD long-form as recitation-track sessions.**
5. **Scoreboard + review page integration.**
6. **Baseline run, set thresholds.** Session measures appear on the scoreboard as *reported*, not gating, beside today's gates.
7. **Recorded sessions**: collection protocol, labelling guide, then compare stitched vs recorded numbers.
8. **Promote to gating** only once: stitched and recorded numbers agree within agreed bounds per track; coverage minimums are met (at least 20 speakers, at least 15 labelled errors per gated class per track, at least 60 clean minutes per track); and both have run side by side for two scoreboard runs. Today's clip error/position gates then become non-gating reports; listener, runtime and lead-in gates are untouched.

## Risks

- **Stitched audio flatters the engine.** Seams can be cleaner than real pauses, and planted errors are read deliberately. Mitigation: step 7, and report stitched and recorded numbers apart until they agree.
- **Too few consecutive same-speaker clips** to build many sessions. Measure coverage in step 2 before committing; if thin, record more clean continuations per speaker rather than mixing voices.
- **Context levels multiply runtime.** Five levels × sessions; listener evidence is computed once per session, so only tracker+checker reruns, which is cheap.
- **Gate churn.** Switching the release gate changes what "ready" means. Covered by step 8's preconditions.

## Decisions

- **Context mix.** Recitation (tracking) sessions mostly start with no context; memorization/correction sessions mostly have some. So the recitation track gates at `none`, and the memorization track gates at `assignment`/`resume`, with `none` reported. Consequence: twin-ayah latching (and stickiness) is a *recitation* problem first: wrong position shown at `none` is that track's key gate.
- **Live alerts.** The SDK surfaces each error as it's confirmed; apps typically alert live (a beep), and the reciter carries on. So in the memorization track, confirmed-warning delay (p90) is a **gate**, and a false warning is an interruption, which keeps precision a gate. Thresholds come from the baseline.
- **Recorded sessions are crowd-sourced.** 30–50 public sessions. Needs: an explicit consent and licence at upload (voice recordings, use in a public benchmark); labels by two reviewers, disagreements resolved by a third, using the existing review page; each contributor capped at a few sessions so no voice dominates; a held-out subset never used for tuning.
