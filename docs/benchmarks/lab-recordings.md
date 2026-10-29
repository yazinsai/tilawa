# Scripted-error recordings

Real recitation with mistakes planted on purpose, so the labels are known by construction and nothing is spliced in (`.scratch/lab/issues/18-scripted-recordings.md`). Spliced deletions sound natural, spliced insertions don't, so the built test sets keep their splices and these takes supply the real wrong words, added words and self-corrections. Omissions are scripted too, at a lower weight.

- Page: <https://amal-invest--tilawa-lab-review-web.us-east.modal.run/record> (live, after the promote below). Preview: <https://amal-invest--tilawa-lab-record-preview-web.us-east.modal.run/record>.
- A reciter opens `/record`, types a name (kept in the browser; `?speaker=<name>` works too), picks a script, reads the steps and the marked page, records, listens back and uploads. The verdict shows on the page about a minute later.
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
