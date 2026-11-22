# Checker false warnings on long stitched sessions (stitched-warnings-v1)

Question from ticket 07: the shipped `full` int8 student gives 2.73 lead-out false warnings per 5 min on the lead-in/lead-out eval (bar ≤ 1). Is that because each help session is short and ends inside the lead-out, or is it the checker?

**Answer: it is the checker.** On ticket 13's 129 clean stitched sessions (747 min), the free-recitation checker confirms about 1.4–1.5 false wrong/added words per 5 min for both listeners. The rate in the last 20 s of a session matches the rate mid-session. Session-end finalisation adds 0 warnings for int8 and 2 for the teacher across 129 session ends.

## Setup

- Sessions: `jumps-v1-manifest.json` (ticket 13): 121 tlog sessions (706 min, clean-match test + val) and 8 help-speaker sessions (40 min, help eval clean clips). All clips are clean, so every confirmed `wrong_word` or `added_word` counts as false. Omissions and skips aren't counted, because clip spans can trim an edge word.
- Evidence: each clip is decoded on its own and stitched at the evidence level, then replayed in 1 s chunks. Free recitation, memorization tracker, no passage. Checker defaults: latching, local-window confirmation, 2.5 s delay.
  - Teacher: the cached `.scratch/jumps/evidence` and `.scratch/help/wav`, both with posteriors.
  - `spliced-clean-int8`: the shipped `full` int8 model on onnxruntime-node (`pnpm --filter @tilawa/sdk decode`, 20 ms pushes, top-5 posteriors). For tlog, the 3,815 clips were fetched with `tools/jumps_build.py --audio` and decoded into `.scratch/jumps/evidence-int8`. For help, the cached `.scratch/sdk-help/spliced-clean-int8-node-lw`. No Modal runs.
- Split: "confirmed by session end" means the warning was confirmed in `checker.end()`, the finalisation step. "Last 20 s, live" means it was confirmed while pushing, within the session's final 20 s. Everything else counts as mid-session. The end region is 129 × 20 s = 43 min. That is close to the lead-out eval's 53 min over 354 lead-outs.
- Run: `pnpm --filter @tilawa/core stitched-warnings --arm <listener>:<tlog evidence dir>:<help evidence dir>`.

## Results

| Listener | Posteriors | Finalise | Min | Warnings (per 5 min) | Mid-session (per 5 min) | Last 20 s, live | Confirmed by session end | End region (per 5 min) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| teacher | off | on | 747 | 311 (2.08) | 296 (2.04) | 13 | 2 | 15 (1.74) |
| teacher | on | on | 747 | 211 (1.41) | 199 (1.41) | 10 | 2 | 12 (1.40) |
| teacher | on | off | 747 | 209 (1.40) | 199 (1.41) | 10 | 0 | 10 (1.16) |
| spliced-clean-int8 | on | on | 743 | 217 (1.46) | 205 (1.47) | 12 | 0 | 12 (1.40) |
| spliced-clean-int8 | on | off | 743 | 217 (1.46) | 205 (1.47) | 12 | 0 | 12 (1.40) |

By source (posteriors on, finalise on):

| Listener | tlog (706 min) | help speakers (40 min) |
|---|---:|---:|
| teacher | 192 (1.36 per 5 min) | 19 (2.35) |
| spliced-clean-int8 | 194 (1.38) | 23 (2.84) |

## Reading

- **`e2e.stitched.falseWords` already counted checker warnings.** It counts confirmed `wrong_word` and `added_word` events from a free-recitation checker. But `pnpm jumps` replays without posteriors, so local-window confirmation never ran. The first row reproduces it: 311 here against 309 at `db72fff`. With posteriors, as the shipped checker runs, the teacher rate is 1.41, not 2.07. The scoreboard now reads this report for both listeners (posteriors on, finalise on), so int8 gets a value for that gate.
- **The session end isn't special.** For both listeners the end-region rate (1.40) equals the mid-session rate (1.41–1.47). Finalisation adds 2 warnings in 129 session ends (teacher) or none (int8). By contrast, the lead-out eval went from 25 to 29 for int8 and from 10 to 36 for the teacher. So the teacher's lead-out jump with finalisation is a short-session artifact: in a 10 s clip plus a 1–2 ayah lead-out, the session end is always near a just-recited, still-suspected word. The int8 lead-out rate is not that artifact. With finalisation off it is already 25 in 53 min (2.35).
- **The int8 lead-out rate matches its long-session rate on the same speakers.** On help-speaker sessions, int8 gives 2.84 per 5 min against 2.73 on lead-outs. Lead-outs mostly come from the same help speakers' clips. On tlog, both listeners sit near 1.37. So the checker has a baseline of about 1.4 per 5 min on clean continuous recitation, and help-speaker audio (phone recordings, more varied) roughly doubles it. The help-speaker numbers rest on 40 minutes (19–23 warnings), so read them as indicative.
- The 201 clips carrying the teacher's 211 warnings show no cluster: no clip has more than 2. That makes tlog label noise an unlikely single cause, though some of these "false" warnings may be real misreadings inside clean-match clips (PER ≤ 0.12 cutoff).

## Verdict

The ≤ 1 per 5 min bar fails on long clean sessions as well, at 1.46 for int8 and 1.41 for the teacher, with nothing concentrated at session ends. The lead-out number is a real checker false-warning rate on help-speaker audio, not a short-session artifact. Work on it belongs in the checker's confirmation step, for example the local-window margin or a confidence floor. Session-end handling isn't the place.

## Limits

- Evidence is stitched per clip, not one decode of continuous audio. Seams are clean and the student never sees context across clips.
- tlog sessions change speaker at most seams (see `jumps-v1.md`).
- The 20 s tail is a fixed window, not the lead-out's exact length (1–2 ayahs).

## Re-run at 441dddc3 (V1 release candidate, 2026-11-11)

Same manifest and evidence (int8 decodes re-verified byte-identical to the frozen listener), engine at the frozen V1 commit `441dddc3`. Evaluation only; the JSON above now holds this run.

| Listener | Posteriors | Finalise | Sessions / min | Warnings (per 5 min) | Mid-session | Last 20 s | At session end | tlog sessions | help-speaker sessions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| teacher | off | on | 129 / 747 | 279 (1.87) | 267 | 10 | 2 | | |
| teacher | on | on | 129 / 747 | 212 (1.42) | 201 | 9 | 2 | 186 in 706 min (1.32) | 26 in 40.5 min (3.21) |
| teacher | on | off | 129 / 747 | 210 (1.41) | 201 | 9 | 0 | | |
| spliced-clean-int8 | on | on | 129 / 743 | 200 (1.35) | 192 | 8 | 0 | 172 in 702 min (1.22) | 28 in 40.3 min (3.48) |
| spliced-clean-int8 | on | off | 129 / 743 | 200 (1.35) | 192 | 8 | 0 | | |

The int8 gate number moves 217 → 200 (1.46 → 1.35 per 5 min), still above the ≤ 1 bar. Help-speaker audio is still about 2.5× the tlog rate. See `v1-release-readiness.md`.
