# Scripted-error recordings

Real recitation with mistakes planted on purpose, so the labels are known by construction and nothing is spliced in (`.scratch/lab/issues/18-scripted-recordings.md`). Spliced deletions sound natural, spliced insertions don't, so the built test sets keep their splices and these takes supply the real wrong words, added words and self-corrections. Omissions are scripted too, at a lower weight.

- Page: <https://amal-invest--tilawa-lab-review-web.us-east.modal.run/record> (live, after the promote below). Preview: <https://amal-invest--tilawa-lab-record-preview-web.us-east.modal.run/record>.
- A reciter opens `/record`, types a name (kept in the browser; `?speaker=<name>` works too), picks a script, reads the steps and the marked page, records, listens back and uploads. The verdict shows on the page about a minute later. A take that failed in a few ayahs can be fixed by recording only those ayahs again (see Fixing a passage).
- Processing: `modal/lab_recordings.py` (app `tilawa-lab-recordings`, `LAB_RECORDINGS_APP` overrides). Pure logic: `tools/lab_recordings_core.py`. Scripts: `tools/lab_scripts.py`.
- Test: `uv run -q --python cpython-3.12-macos-aarch64-none --with quran-transcript==0.6.4 --with rapidfuzz==3.14.1 --with numpy --with pyarrow --with pytest pytest tools/test_lab_scripts.py tools/test_lab_recordings.py`

## Scripts

`tools/lab_scripts.py batch` writes `docs/benchmarks/lab-scripts-v1.json` (batch `rec-v1`, seed 20261029). A script is a passage of 3–6 minutes at a normal pace, from surahs a reciter is likely to know (Juz Amma, Juz Tabarak, and Al-Kahf, Ya-Sin, Ar-Rahman, Al-Waqi'a, Maryam, Al-Baqara). Its mistakes are planted about one per 2–3 ayahs, never in the first two ayahs or the first 20 s, and at least 8 words and 10 s apart.

- **Wrong word** (weight 0.5). The substitute is, in order: the twin ayah's word, a word that sits between the same neighbours elsewhere (`before`, `after`, `context`), or a look-alike word (`confusable`). Each carries `forms_other_ayah` by the `tools/help_import.py` rule.
- **Added word** (0.3, at most 35% of a script's mistakes): a word from elsewhere in the passage, said before a word.
- **Omitted word** (0.2).
- **Self-correction** (1–2 per planted script): say a wrong word, stop, go back and say it right. Labelled as a `self_correction` non-error, not a mistake.

Every word records what is to be said. An ayah with a plant is phonetized again as said (quran-transcript 0.6.4, ref-v2 profile), so the planted words carry their phonemes in context. The isti'adha before and صدق الله العظيم after are optional, unscripted items built from the Quran's own words. The basmala is scripted where a surah starts.

| Script | Passage | Min | Ayahs | Plants |
|---|---|---|---|---|
| rec-v1-01 | Ad-Duha–Al-Bayyina 93:1–98:8 | 5.2 | 59 | wrong 7, added 4, omitted 4, self-correction 1 |
| rec-v1-02 | Nuh 71:1–28 | 3.8 | 28 | wrong 4, added 2, omitted 3, self-correction 1 |
| rec-v1-03 | Al-Baqara 2:1–24 | 4.8 | 24 | clean |
| rec-v1-04 | Al-Inshiqaq–Al-Buruj 84:1–85:22 | 3.8 | 47 | clean |
| rec-v1-05 | Al-Insan 76:1–31 | 4.4 | 31 | wrong 5, added 2, omitted 4, self-correction 1 |
| rec-v1-06 | Maryam 19:1–32 | 5.0 | 32 | wrong 6, added 2, omitted 2, self-correction 2 |
| rec-v1-07 | At-Takathur–Al-Kafirun 102:1–109:6 | 3.6 | 45 | wrong 5, added 3, omitted 2, self-correction 1 |
| rec-v1-08 | Al-Muddaththir 74:1–56 | 4.3 | 56 | wrong 7, added 4, omitted 2, self-correction 1 |
| rec-v1-09 | Ar-Rahman 55:1–42 | 3.7 | 42 | wrong 4, added 3, omitted 3, self-correction 1 |
| rec-v1-10 | Al-Infitar–Al-Mutaffifin 82:1–83:36 | 4.5 | 55 | wrong 8, added 4, omitted 4, self-correction 1 |

43.1 estimated minutes and 419 ayahs. There are 94 mistakes (46 wrong, 24 added, 24 omitted) plus 9 self-corrections, and 3 mistakes form another ayah. Wrong-word substitutes: 36 context, 24 confusable, 15 after, 2 before, 2 twin.

Publish a batch (the page lists every `recordings/scripts/*.json`): `modal run modal/lab_recordings.py --action scripts --file docs/benchmarks/lab-scripts-v1.json`. Republishing the same batch keeps old takes. Each take records the script sha it was recorded against.

## Processing a take

The upload is saved as `recordings/<speaker>/<script>/take-<n>.<ext>` (webm, mp4, ogg or wav, as the browser records it) with its metadata. Then `process_take` is spawned:

1. ffmpeg converts it to `recordings/sessions/<id>.phone.wav`, 16 kHz mono PCM16, between 5 s and 15 min. `<id>.checks.json` holds its sha256 (`phoneSha256`), as for the built sets.
2. The teacher (`obadx/muaalem-model-v3_2`, the `lab_decode` settings) writes `<id>.teacher.phone.evidence.json`.
3. `check_take` CTC-aligns what the script says was said to the teacher's posteriors (`lab_checks`):
   - **Heard**: the teacher's greedy phonemes over the script's. Below 0.5, the take is silent or a different passage: **redo**.
   - **Optional items**: an isti'adha, closing or basmala heard with PER above 0.3 wasn't said, so it is dropped and the alignment is re-run. Unscripted items are dropped first, at most 4 passes.
   - **Each plant**: the log-likelihood ratio of the planted words against the original text at that spot, the greedy decode's edit distance to each, and the local PER. LLR ≤ −3 nats is **not performed** (the reciter read it as written). Otherwise, LLR under 3, a decode no closer to the plant, or local PER above 0.25 is **unsure**, and the rest is **performed**.
   - **Each ayah**: PER of the greedy decode against what was to be said. Above 0.15, the ayah is flagged as holding an unscripted change.
   - **Coverage**: the share of the script's words found.
4. Verdict:
   - **redo**: a plant not performed, coverage under 90%, 3 or more flagged ayahs, alignment failed, or too little heard.
   - **listen**: an unsure plant or a flagged ayah.
   - **ok**: otherwise.

   The page shows why, per plant.
5. A take that isn't redo gets a BenchSession (`recordings/sessions/<id>.json`, validated against the core rules before it's written):
   - `source: "recorded"`, context none;
   - word times from the alignment;
   - wrong and added words carry their `said` phonemes;
   - an omission is an event in the gap;
   - a self-correction is a `self_correction` non-error;
   - the isti'adha and closing are `non_quran` spans labelled with their role.
6. `rebuild` rewrites three files:
   - `recordings/review/takes.json`, the spot-check queue in the ticket 06 format. It holds every unsure plant and flagged ayah, plus per take a seeded 3 performed plants, 1 self-correction and 2 clean 8 s windows. It appears in the review page's queue list.
   - `recordings/manifest-v1.json`: the latest non-redo take per speaker and script, so recording again replaces a take.
   - `recordings/index.json`.

`--action process --speaker <s> [--script <id>] [--take <n>]` re-runs the checks, and so does the page's retry button. A failed run writes verdict `error` with the exception.

## Fixing a passage

A take that is redo or listen because of a few ayahs can be fixed by recording only those ayahs again. The joined result is a new take, checked end to end like any other.

1. **What to fix.** Under the newest take, the page lists the failed ayahs, from its checks:
   - a plant not performed or unsure;
   - an ayah flagged as an unscripted change;
   - words the alignment didn't find (coverage gaps).

   Failed ayahs at most 2 ayahs apart are one passage, and the page offers one passage at a time. The "From" and "to" fields widen it. Fixes are whole ayahs only. A basmala goes with ayah 1 and an added word with its ayah. The isti'adha and the closing stay with the take. A take that didn't align at all gets no offer: record it again.
2. **Record.** "Re-record ayahs A–B" puts the recorder in fix mode. The passage is marked in the text, and the dock reads "Fix for take N". Pause and resume work as in a full take. The upload holds the next take number at once: `take-<m>.patch.<ext>`, with `take-<m>.json` carrying `fix: {base, from, to, patch, resumesS}`.
3. **The patch's own check.** `process_take` sees `fix` and calls `compose_fix`:
   - the patch is converted (1 s to 15 min) and decoded by the teacher to `sessions/<id>-patch.*`;
   - `check_take` runs on the script cut to A..B (`fix_script`), so the patch's plants get the same LLR, decode and local-PER checks;
   - a plant on the patch's first or last word is checked against the word on its one side (`plant_window(open_ends=True)`), and full takes are unchanged;
   - a patch whose own verdict is redo is refused.
4. **Joins.** `plan_fix` finds four pauses:
   - in the take, the gap between the last word before A and A's first word (searched 0.1 s into the word before and 0.05 s into A, for alignment slack), and the gap after B;
   - in the patch, the silence before its first word and after its last.

   Each must hold a run of silent 10 ms frames at least 80 ms long. A frame is silent at or below the floor from the lab_sessions weak-onset rule: room tone (the 2% quietest frame) + 3 dB, or 30 dB under the speech level where the room is quieter than that. The floor is capped at a pause's 12 dB under speech (`PAUSE_DB`), so a noisy room has no silence rather than silence everywhere. The cut sits mid-pause, keeping at most 0.4 s of pause on the side of the speech that is kept, with its 20 ms crossfade inside the silence. In the take, the first such pause after the word before A is used, and the last one before the word after B. In the patch, the last one before A and the first one after B.
5. **Level.** The gain is the take's speech level (energy mean of the loudest half of its frames) minus the patch's, between its cuts. It is applied to the patch and reported. A gap over 6 dB is refused.
6. **Stitch.** The composed take is the take up to the cut before A, then the patch times the gain, then the take from the cut after B. It is written as `take-<m>.wav`. `take-<m>.json` gets:
   - `joinsS`, the two join times on the composed clock;
   - `resumesS`: the base take's resume points outside the replaced passage, the patch's inside the part kept, and the two joins;
   - `fix.joins` (take and patch times, the pause length heard at each join, the silence measured on each side) and `fix.gainDb`.
7. **Full checks.** The composed take is processed end to end: whole-script alignment, every plant, coverage and flagged ayahs. The session's meta carries `joinsS`, `resumesS` and `fix`. The review queue gets one clean "Join of a fixed passage" item per join. The manifest keeps the latest non-redo take, so a good composed take replaces the one it fixed. The base take and the patch stay on the volume, and a composed take can be fixed again (take 5 below fixes take 2).

A fix that can't be joined gets verdict **refused** ("Could not join") with the reason and, where it helps, the passage to record instead. The page offers that passage as a button:

| Case | Message | Offered |
|---|---|---|
| No silence in the take before A (the reciter ran into A) | no clean pause before A in take N: record again from A−1 | A−1..B |
| No silence in the take after B | no clean pause after B in take N: record again up to B+1 | A..B+1 |
| A is the script's first ayah or B its last, and that side fails | … record the whole take again | none |
| No silence at the start of the patch | no silence before A at the start of your fix: after you start recording, wait a moment before you recite | A..B |
| No silence at its end | no silence after B at the end of your fix: pause for a moment before you stop | A..B |
| The patch's own check is redo (silent, other text, plant not performed) | the fix itself: …; record A..B again | A..B |
| Level mismatch over 6 dB | your fix is X dB quieter/louder than take N: record it where you recorded the take, as far from the microphone, or record the whole take again | A..B |
| The base take isn't checked and aligned, or is from an older script version | refused at upload (HTTP 409) | none |

## Fixing a passage

A take that fails in a few ayahs doesn't have to be recorded again in full. On the newest take, if it is redo or listen and it aligned, the page lists the ayahs that failed, with why:

- a planted mistake not heard or not sure;
- an ayah that sounds different from the script;
- words the alignment didn't find.

Failed ayahs at most 2 ayahs apart form one passage. The page offers **Re-record ayahs A–B**, defaulting to the first passage. Each passage in the list sets the range, and the two selects widen it, whole ayahs only. One fix covers one passage; the next fix starts from the joined take. The marked ayahs are highlighted in the text, and the recorder records only them, with pause and resume as for a full take. Then the patch is uploaded.

A fix is a new take number reserved at upload. Its files are `take-<n>.json` (with `fix: {base, from, to, patch, resumesS}`) and `take-<n>.patch.<ext>`. `process_take` composes it first (`compose_fix`):

1. Convert the patch, decode it with the teacher (`sessions/<id>-patch.*`), and run `check_take` on what the script says for A..B (`fix_script`; a plant on the patch's first or last word is judged against the one word beside it). If the patch alone is redo, the fix is refused.
2. Find the joins (`plan_fix`). In the base take, the pause between the word before A and A's first word, and the pause after B's last word. In the patch, the silence before its first word and after its last. Each search reaches 100 ms into the aligned word before and 50 ms into the word after.
3. Every join needs measured silence: at least 80 ms of 10 ms frames below the recording's silence floor. The floor is the lab_sessions weak-onset rule: room tone (2nd-percentile frame) + 3 dB, or 30 dB under the speech level when the room is quieter than that. It is never above 12 dB under speech, so a noisy room has no silence rather than silence everywhere. The cut is mid-pause, keeping at most 0.4 s of each side's pause, and the 20 ms crossfade lies inside the silence.
4. Match levels: speech level (energy mean of the loudest half of frames) of the base take against the patch between its cuts. The gain is applied and reported. A gap of more than 6 dB is refused.
5. Stitch the base take up to the join before A, the patch, and the base take from the join after B, then write `take-<n>.wav`. Then `take-<n>.json` gets `joinsS` (the two joins on the composed clock), `resumesS` and `fix.joins` / `fix.gainDb`. `resumesS` is the base take's resumes outside the replaced passage, the patch's resumes inside it, and the joins.

The composed take is then processed end to end like any take: whole-script alignment, every plant, coverage and flagged ayahs. The base take and the patch are kept. The manifest still takes the latest non-redo take, so a good fix replaces the failed take. A refused fix has verdict `refused` and is never current. Its BenchSession `meta` carries `fix`, `joinsS` and `resumesS`. Each join is a spot-check item in `review/takes.json` ("should sound like one take").

The page shows refusals with what to do. A refusal on the take's side comes with a wider range to record, and one button starts that fix:

| Refusal | Says |
|---|---|
| no silence in the take before A | no clean pause before A in take n: record again from A-1 (the whole take when A is the first ayah) |
| no silence in the take after B | no clean pause after B in take n: record again up to B+1 |
| no silence at the patch's start or end | after you start recording, wait a moment before you recite / pause for a moment before you stop |
| level | your fix is x dB quieter/louder than take n: record it where you recorded the take, as far from the microphone, or record the whole take again |
| the patch's own check is redo | the fix itself: (its reasons); record A–B again |
| base take not aligned, or from an older script version | record the whole take again |

### End-to-end check (preview, speaker `test`, 2026-10-30)

`--action test-fix` cuts takes of `rec-test-77` from an mp3quran reader with 0.35 s of silence at every ayah end. The full take reads mistake 1 (77:3) as written and runs 77:13 into 77:14 with the source's own (reverberant) pause. The patches perform the plants of their ayahs.

| Take | What | Verdict | Notes |
|---|---|---|---|
| 1 | full take | redo | mistake 1 not performed (LLR −34.5). Offers 77:3, and 77:14 (omission unsure). |
| 2 | take 1 + 77:3 | ok | Patch: mistake 1 performed (LLR +29.9). Joins at 8.8 s and 13.1 s with a 0.6 s pause each. Gain +0.9 dB. All 5 plants performed. |
| 3 | take 1 + 77:3, no lead silence | refused | no silence before 77:3 at the start of your fix |
| 4 | take 2 + 77:14 | refused | no clean pause before 77:14 in take 2: record again from 77:13 (offered) |
| 5 | take 2 + 77:13–14 | ok | Fix of a fix. Gain −1.1 dB. |
| 11 | take 10 (= take 1) + 77:3 recorded in Chrome through the page (fake microphone) | listen | Mistake 1 performed. The omission at 77:14 is unsure again (local PER 0.29), and the page offers it next. |

On take 11's composed WAV, the audio within 40 ms of each join is at −83 dB (speech −18 dB), with no sample step. The manifest picked take 11, and the queue held its two join items. A silent browser recording (Chrome's fake microphone needs `--no-sandbox` and a `/private/tmp` path) was refused as "the fix itself: the teacher hears almost none of the script". Uploading the same full take twice gave slightly different teacher LLRs and once one more flagged ayah (77:12), from GPU type. All test data was deleted afterwards (`--action cleanup --speaker test --batch test`).

A speaker's split comes from `recordings/splits.json` (`{"<speaker>": "exam"}`), default practice. The manifest is phone-only (`variants: ["phone"]`) and every entry carries `tags: ["recorded"]` and `slices.source: "recorded"`.

## Joining a test set

```sh
modal run modal/lab_recordings.py --action join --base lab-v1.4 --name lab-v1.4r
```

This writes a new set at `/lab/data/lab-v1.4r/`. The base's audio, evidence, sessions and checks are copied, then the accepted takes' files are copied into `sessions/` beside them (ids start `rec-`, so the evidence pattern finds them). The manifest is the base's plus the recorded entries, with `joinedFrom` and `recorded: {from, sessions}`. The base is untouched. The name must be new (test set versions are immutable), and an id clash fails.

- Recorded sessions have teacher evidence only. Decode the candidate before scoring: `modal run modal/lab_decode.py --listener <student> --model <dir> --test-set lab-v1.4r`. Phone-only sessions skip the studio variant, and existing evidence is kept as fresh.
- `lab-cli` reads `slices.source`, and the report has `slices.recorded.{practice,exam,all}` on phone audio beside `studio` and `strict`. Recorded sessions also count in the headline split they belong to.

## End-to-end check (preview, speaker `test`, 2026-10-29)

Al-Mursalat 77:1–20 (`tools/lab_scripts.py e2e`) has 5 plants: a wrong word at 77:3 and at 77:18, an added word before 77:7, a self-correction at 77:10 and an omission at 77:14. There is also a clean copy. `--action test-audio` cut takes from an mp3quran reader as webm/opus: once performed as scripted, once as written. They were uploaded through `/api/record/upload`, and one more was recorded in Chrome through the page with a fake microphone.

| Take | Verdict | Why |
|---|---|---|
| scripted, performed | listen | 4 plants performed (LLR +18 to +85). The omission at 77:14 is unsure (local PER 0.29): the cut audio says مَاا where the script expects مَ |
| scripted, read as written | redo | all 5 plants not performed (LLR −35 to −144), 5 ayahs flagged |
| clean | ok | |
| clean, through the page | listen | 77:20 cut off: recording stopped at 82 s of the 84.6 s source |

On both uploaded takes the isti'adha and closing were dropped as unsaid, and heard was 1.06. The core validator (`lab-validate-cli --dir`) passed the sessions and the manifest. A join into a throwaway set (`lab-v1.4` + 2 takes, 248 files) passed the validator. Scoring the 2 takes plus one base session with the teacher as listener gave a `recorded` slice of 2 sessions with 4 mistakes. The engine caught 3: one added word misclassified, and the wrong word at 77:18 (لِّلْمُكَذِّبِينَ from 77:15 for بِٱلْمُجْرِمِينَ) missed by the checker even on oracle evidence. The test takes, queue entries and the throwaway set were deleted afterwards (`--action cleanup --speaker test --batch test`).

No scoreboard row: no metric moves until a joined set is scored.
