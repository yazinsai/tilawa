# 20 Bigger synthetic test set

Status: ready-for-agent
Type: task
Blocked by: none

Each half of the current test set (lab-v1.4.v3) has about 88 recited minutes and about 134 planted mistakes, so a count of 14 wrong calls wobbles by about ±4 from which sessions happen to be in it. Per class it is thinner: practice has only 9 wrong words, the most important class. Experiments moving practice by 5 can't be told from noise.

Build the next test set version with about 4–5 times the sessions per half, using the existing spliced session builder (ticket 03, `modal/lab_build_sessions.py`) and its rules (lab-v1.2 and later).

## Notes

- Only 32 held-out reciters exist (16 practice, 16 exam; `docs/benchmarks/lab-reciters-v1.json`). More sessions means more passages per reciter, not more voices. Keep each reciter in its split.
- Weight the plant mix toward wrong words, with mutashabihat swaps included, aiming at 100 or more of each class per half.
- Keep the current sessions as a subset, so the new version's numbers can be compared on the old sessions too.
- Flags from earlier versions (`flags-v3.json`) must carry over to the sessions they apply to.
- Decoding and scoring the larger set cost Modal time. Check the experiment run time, about 11 min today, set by the longest session.

## Acceptance

- The new version validates, renders, and passes the builder's seam and render checks.
- A spot-check listening queue goes to the user, like ticket 07's v1 check, before the set is trusted.
- `lab apply-flags`-style registration as the current test set, with the main line re-scored on it.
- Write-up with counts per class and split, and the main line's practice numbers on the new version next to the old.

## Comments

Progress (30 Oct). The new version is **lab-v1.5**: lab-v1.4's 40 sessions frozen byte for byte, plus 80 new sessions per split, for 100 per split (5×). Every lab-v1.2–v1.4 rule is unchanged (7e7a81c, 083ba6e).

- **Frozen subset.** `seed_frozen` copies each lab-v1.4 session into `/lab/data/lab-v1.5/sessions/` under its old id (`lab-v1.4-p01`…). That covers the label, both WAVs, render checks and every listener's evidence, plus their decode records and refusals. The planner reads them back and never replans them. They take their share of the slices and clean slots, and no new window overlaps them. lab-v1.4's applied flags (`flags-v3.json`, 4 windows) become `lab-v1.5/flags-v1.json` with the same sessions and times.
- **Supply first** (`--stage supply`). Across the whole held-out pool, only a few words have a site under the pause-to-pause and host-recording rules:
  - practice: 100 mutashabihat-swap words in 25 recordings and 229 wrong-word words in 84 (259 ayahs);
  - exam: 109 in 26 and 238 in 75 (264 ayahs).
  - Surah 26 holds a third of practice's sites. 100 wrong-class plants per half therefore needs windows placed on the sites.
- **Planner changes, all for new sessions only:**
  - A recording may host several sessions in non-overlapping windows. Before, it was one window per reader and surah.
  - A session with mistakes ranks its windows by how many ayahs hold a wrong-word or twin site (up to `SITE_CAP` 3). A clean session takes the fewest.
  - A normal-start session with mistakes may move to the split's reader with the richest window, up to `READ_SLACK` 3 sessions over an even share.
  - A wrong-word site always wins its ayah in the one-per-2–3-ayahs loop, and the loop leaves room when the next ayah holds one.
  - Slices are scaled ×5: per split 20 ayah-1, 15 strict, 15 mutashabihat focus and 25 clean.
- **Dry run.** Per split: 100 sessions, about 445 min, sessions of 180–416 s, as before.
  - practice: wrong 37, mutashabihat 52, omitted 313, added 134, skipped 134;
  - exam: 38, 52, 249, 112, 127.
- **Deviation: the wrong class reaches about 90 per half, not 100.** That is up from 9 on practice. The windows hold 89 and 97 site ayahs, and the edit spacing and ayah-edge rules lose the rest.
  - `SITE_CAP` 6 or 10 and a higher mutashabihat cap per session change nothing.
  - Going further needs a rule change, which is your call: donors from the reader's other recordings, or inserted audio at linked seams. Alternatively, more than 5× the sessions.
- **Reader spread:** practice readers have 5–8 sessions each and exam readers 3–10, against 1–2 before.
- `tools/lab.py` now gives one scoring shard per session (`MAX_SHARDS` 200), so a run should still take about as long as its longest session.
- The build loop is running: plan, render, gate, teacher, checks, refusals, then `spliced-clean-int8` and triage.

Built (31 Oct). Write-up in `docs/benchmarks/lab-v1.5.md`, counts in `lab-v1.5.json`.

- **Loop.** Converged in 12 iterations over three runs. The first stopped at a `lab_check` bug, fixed in 15ec94d: its queue check covered frozen sessions. The second stopped at the iteration cap.
  - Refusals moved windows, so later iterations replanned many sessions. New refusals per iteration fell to 0.
  - No frozen session failed any studio check.
- **Acceptance: validates, renders, passes seam and render checks.**
  - Core `validateSession`: 200 files and 200 manifest copies, 0 failures.
  - Seam gate: 1734/1734 seams. 43,066/43,066 words found, 0/320 clipped words over the limit, 0/384 donors short, all `spliceRules` 0.
  - Studio checks: 1150/1150 mistakes, 200/200 non-errors, 1731/1731 clean stretches, 200/200 timings.
- **Counts** (practice / exam):
  - sessions: 100 / 100; minutes: 434.2 / 434.3; clean: 26 / 26;
  - wrong word: 31 / 37; mutashabihat: 48 / 49; omitted: 298 / 220; added: 118 / 101; skipped: 122 / 126.
  - **Deviation:** the wrong class has 79 / 86, not 100. The loop's refusals took about 10 per split off the dry run. Going further needs a rule change (see above). Practice had 9 before.
  - **Biases:** surah 26 hosts 51 of 200 sessions, and readers recite 3–10 sessions each.
- **Triage.**
  - Nemotron + Soniox: 7285 of 7580 auto-accepted, 0 both-mismatch, 19 disagree and 276 unsure.
  - The full queue `review/triage.json` has 370 items, with maxBad 9 and 66 lab-v1.4 verdicts carried over.
  - Soniox cost $1.68. lab-v1.4's transcripts were seeded for the frozen clips, and the `soniox` function now runs on 8 containers.
- **Spot check: `lab-v1.5/review/spot-check.json`, 45 items, maxBad 1, waiting for your listen.** It holds every disagreement plus seeded unsure, seam (inserted first) and auto-accepted items, from new sessions only (`tools/lab_spot_check.py`).
- **Main line on practice** (15ec94d; scored directly, ledger untouched):
  - all 100 sessions: 107 wrong calls (82 missed + 25 false warnings) over 429.5 min, 1.25 per 5 min;
  - the frozen 20: 11 (11 + 0), identical to lab-v1.4.v3's 11 (11 + 0);
  - the new 80: 96 (71 + 25), 1.41 per 5 min against 0.62 on the frozen 20.
  - Either the folded experiments fit the old 20 practice sessions, or the new sessions hold bad data. The spot check is the first test.
  - Run time: 565 s wall on 100 shards (one per session).
- **Not done yet: registration.** `lab baseline --manifest lab-v1.5/manifest-v1.json --exclude lab-v1.5/flags-v1.json` would make lab-v1.5 the ledger's current test set and re-score the main line on practice and exam. It waits for the spot check, since the set isn't trusted before then. No experiments are open.

Decision (31 Oct, you): the wrong-class shortfall (79 / 86 per half) is accepted, with no rule change and no more sessions. lab-v1.5 is registered once `review/spot-check.json` passes (at most 1 bad in 45). The spot check has 0 of 45 reviewed so far.

Review page in plain words (31 Oct, after your note that the spot-check wording was unreadable). A reviewer subagent listed every confusing string. `tools/lab_plain.py` (tested in `tools/test_lab_plain.py`) now builds each item's heading from the item itself, so every existing queue reads this way without being rebuilt. Ids, clips and verdicts are unchanged.
- **Heading:** the reason as a tag, then the listening question ("Listen for: «بِمَا» should be missing between «أَعْلَمُ» and «تَعْمَلُونَ».").
- **Location line:** surah, ayah and word, the time into the clip (not into the session), and how the cut was made.
- **Arabic text:** a dashed, struck-through gap where words were left out, and a mark over the whole of a changed, extra or repeated phrase.
- **Panels** ("What the computer checks think", "Sound by sound"):
  - They now sit below the verdict buttons.
  - The models are "Check 1" and "Check 2", each with one plain verdict, and the numbers are on a small grey line in words.
  - The sound-matching check's rough write-down is hidden behind a toggle.
  - Made-up comparison versions for clean windows are labelled as made up.
  - The listener summary names the word and says whether what was heard is expected.

`/api/queue` adds `plain` to each item. `lab_models_core` and `lab_phonemes_core` were reworded and their tests updated. Deployed to `tilawa-lab-review`.
