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
