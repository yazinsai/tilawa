# Stitched-session jump test jumps-v1 (ticket 13)

Report jumps-v1, commit 7e91942, reference ref-v2. 129 sessions, 747 minutes, stitched from clean held-out clips (manifest `jumps-v1-manifest.json`, built by `tools/jumps_build.py`). Clips are joined with 0.3–1.5 s gaps of blank frames or the clip's own lead-in (room tone).

Teacher evidence is the offline Muaalem decode of each clip, stitched at the evidence level and replayed in 1 s chunks. It is not a streaming student, and no session audio was decoded as one recording.

A jump is a non-local lock change: to an ayah more than 2 from the previous lock. It is false when the new ayah is more than 2 from every ayah recited in the last 4 s (or starting within 0.5 s), as in `acceptance-v3`. A planted jump is a seam whose next clip starts more than 2 ayahs from where the last one ended; it is followed by the first lock on the new run's ayahs before the next planted jump, timed from the new clip's first decoded phoneme. The start is the first lock on the opening run within 5 s of its first phoneme.

## Results

| Measure | All | Contiguous (tlog) | Planted jumps (tlog) | Hard runs (tlog) | Help speakers | Any session with 3+ near-duplicate segments | Bar |
|---|---:|---:|---:|---:|---:|---:|---|
| Sessions / minutes | 129 / 747 | 84 / 554 | 35 / 146 | 2 / 6 | 8 / 40 | 9 / 77 |  |
| Seams / speaker changes | 3861 / 2952 | 2920 / 2278 | 737 / 643 | 37 / 31 | 167 / 0 | 597 / 413 |  |
| Near-duplicate segments | 98 | 75 | 7 | 4 | 12 | 50 |  |
| False jumps / jumps | 3/235 | 1/32 | 0/54 | 0/3 | 2/146 | 2/20 |  |
| False jumps per jump | 1.3% | 3.1% | 0.0% | 0.0% | 1.4% | 10.0% | ≤ 2% |
| Sessions with a false jump | 2/129 (1.6%) | 1/84 (1.2%) | 0/35 (0.0%) | 0/2 (0.0%) | 1/8 (12.5%) | 1/9 (11.1%) |  |
| False jumps per session | 0.02 | 0.01 | 0.00 | 0.00 | 0.25 | 0.22 |  |
| False jumps per 5 min | 0.02 | 0.01 | 0.00 | 0.00 | 0.25 | 0.13 |  |
| Free-recitation checker: false wrong/added words (per 5 min) | 305 (2.04) | 236 (2.13) | 47 (1.61) | 0 (0.00) | 22 (2.72) | 44 (2.84) | ≤ 1 per 5 min |
| Planted jumps followed | 192/216 (88.9%) | 0/0 | 48/48 (100.0%) | 2/2 (100.0%) | 142/166 (85.5%) | 14/18 (77.8%) |  |
| Followed within 5 s | 153/216 (70.8%) | 0/0 | 35/48 (72.9%) | 1/2 (50.0%) | 117/166 (70.5%) | 7/18 (38.9%) |  |
| Follow latency median / p90 | 3.4 s / 6.4 s | — / — | 3.4 s / 6.7 s | 7.7 s / 7.7 s | 3.4 s / 5.9 s | 5.4 s / 6.6 s |  |
| Correct ayah within 5 s of session start | 111/129 (86.0%) | 71/84 (84.5%) | 32/35 (91.4%) | 2/2 (100.0%) | 6/8 (75.0%) | 8/9 (88.9%) | ≥ 95% |

## False jumps

| Session | Time | From | To | Being recited |
|---|---:|---|---|---|
| tlog-contiguous-065 | 31.2 s | 48:29 | 48:26 | 49:1, 49:2 |
| help-help-127 | 145.9 s | 100:7 | 3:2 | 44:1, 1:2, 1:3, 1:4 |
| help-help-127 | 205.6 s | 56:44 | 53:26 | 92:13, 10:84, 10:85, 10:86 |

## Limits

- The ≤ 2% bar in `acceptance-v3` is per clip. Here it is shown per session and per jump; a 3–10 minute session with no false jump is a much stricter test than a 10 s clip with none.
- **Speakers.** tlog has no speaker IDs, and its clusters are coarse. Held-out clips rarely cover consecutive ayahs within one cluster: the longest same-cluster run is about 3.5 minutes. So tlog sessions keep the previous clip's cluster when one is there and change speaker at most seams (the speaker-change row). Teacher evidence is decoded per clip, so to the tracker a speaker change is only a change in evidence quality. A student decoding stitched audio would see more. Help sessions are single-speaker.
- **Contiguity.** A contiguous session steps 1 or 2 ayahs (at most one ayah missing, which is local). Clip spans are the tlog-v2 labels, which can trim an edge word. The tlog pool is the clean-match test and val splits, both held out from student training by cluster.
- **Help sessions** are each eval speaker's clean clips in recording order, 2–10 minutes, for speakers with at least 2 minutes. Nearly every seam is a planted jump, often to a single short ayah: an ar-Rahman refrain ayah, 113:x or 44:1 حم can't be located from its own words, and most unfollowed jumps are of that kind.
- **Hard cases.** Most near-duplicate ayahs (exact phoneme duplicates more than 2 ayahs apart, plus 79:39 / 79:41 and al-Kafirun) fall inside long contiguous runs, so only two dedicated hard sessions were built. The last column slices every session with 3+ near-duplicate segments.
- Stitch seams are cleaner than real pauses: no breath, restart or hesitation. A pass here is necessary, not sufficient; real long sessions (ticket 05) remain the final check.
- Only teacher evidence is scored. Stitched help audio is written under `.scratch/jumps/audio/` for student runs; tlog audio is not local, so tlog sessions have no stitched audio yet.
