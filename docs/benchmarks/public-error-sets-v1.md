# Public error sets v1

Lab ticket 14, spec decision 20. Three public learner-error sets are on the `tilawa-lab` Modal volume. **Nothing is scored.** They wait for the pronunciation track, and RetaSy's wrong-verse labels later become a tracker slice. No metric moves, so there is no scoreboard row.

    modal run modal/lab_public_sets.py [--only ikhlas,retasy,quranmb]
    uv run --python cpython-3.12-macos-aarch64-none --with modal --with pytest pytest tools/test_lab_public_sets.py

All downloads and conversion run on Modal. Each dataset directory on the volume has:

- `orig/`: the source audio as shipped.
- `wav16k/`: 16 kHz mono s16 WAV from ffmpeg.
- `manifest.json`: provenance, counts, issues, and `items`.

Item paths in the manifest are relative to the dataset directory. Every item has `id`, `wav`, `orig`, `duration_s`, `pcm_sha256` (hash of the 16 kHz PCM), `dup_of` (canonical id when the PCM is identical to another item, else null), `issues`, and the normalised label fields listed per dataset below.

| dataset | volume path | rows | unique audio | licence |
|---|---|---|---|---|
| Al-Ikhlas error detection | `/data/public/ikhlas/` | 1,506 | 1,499 | CC BY 4.0 |
| RetaSy quranic_audio_dataset | `/data/public/retasy/` | 6,828 (1,228 labelled) | 6,098 | none stated |
| QuranMB v2 | `/data/public/quranmb_v2/` | 1,642 | 1,640 | none stated; ground truth joined (1,642/1,642) |

## Al-Ikhlas error detection (Mendeley)

**Source.** Mendeley Data `sxtmmr6mvk` v1, doi `10.17632/sxtmmr6mvk.1`, Umm Al-Qura University, published 2024-05-07. File `Dataset and Sounds.zip`, 439,820,286 bytes, sha256 `de8042ac8cc137e83fdcdbed375a24bd51f873c94f7f2cca1e39c60b188c0068`. The hash is verified on every run, and Mendeley's API reports the same hash. According to the description, the audio was collected August 2021 via Google Forms and "Qur'anic audio websites", then labelled by 5 Qur'an experts. The zip, `Dataset.xlsx` and the API record (`mendeley.json`) are kept in `raw/`.

**Licence.** CC BY 4.0, taken from the Mendeley API `data_licence` field. Attribution is required if we ship anything derived from it.

**Counts.** The third-party numbers are confirmed for rows and the label split. "~1,500 unique audio" is really unique *filenames in the sheet*; unique audio after deduplication is 1,499.

| | |
|---|---|
| sheet rows | 1,506 |
| `True or False` = 1 (error) / 0 (correct) | 851 / 655 |
| WAVs in the zip | 1,506 |
| unique filenames referenced by the sheet | 1,500 as shipped, 1,506 after repair (below) |
| unique audio, byte hash / 16 kHz PCM hash | 1,500 / 1,499 |
| rows per verse 112:1 / 2 / 3 / 4 | 378 / 380 / 384 / 364 |
| `ID<n>` numbers (nominal speakers) | 384 |
| total audio | 77.8 min, longest clip 18.8 s |
| source formats | 44.1 kHz stereo 919, 48 kHz mono 378, 48 kHz stereo 145, 44.1 kHz mono 40, 25 others at 16–32 kHz; all 16-bit PCM WAV |

**Label schema.** One sheet, `Sheet1`, with these exact column names (note the trailing space on `Error explanation `):

| column | content | manifest field |
|---|---|---|
| `ID` | row number 1–1506 | `sheet_id`; `id` = `ikhlas-%04d` |
| `Surah name` | always `Surah Al-Ikhlas` | — |
| `Verse number` | 1–4 | `ref` = `112:<n>` |
| `Verse location` | filename `ID<speaker>V<verse><T\|F>.wav` | `file` (after repair), `sheet_file` (only on repaired rows), `speaker` |
| `True or False` | 1 = error, 0 = correct (the reverse of the filename's T/F) | `label`: `error` / `correct` |
| `Error type` | comma list of `تجويد` / `حركات` / `حركات وتجويد`; 0 when correct | `error_classes`: `tajweed` / `harakat` / `harakat+tajweed` |
| `Error location` | comma list of Uthmani words, aligned with `Error type` | `error_words` |
| `Error explanation ` | free Arabic text, comma or `،` separated | `explanations` |
| `Error number` | integer code with no codebook shipped (5 = qalqalah of د, 15 = two qalqalah, 35 = ḍamma on ف + qalqalah, 40 = qalqalah of د in يلد, judging by co-occurrence) | `error_code` |

Error content: 1,055 tajweed entries, 193 ḥarakāt and 13 both. The explanations are dominated by qalqalah of د (944), then ḍamma on ف in كُفُوًا (141, plus 22 with tanwīn), qalqalah in يَلِدْ (31), iẓhār of م (19), sukūn on م (14), kasra on ل in يَلِدْ (13), tanwīn (11), and makhraj of ه / ل (17). Words: أحد 404, يلد 202, الصمد 198, يولد 194, كفوا 176. **There is not one memorization error** (wrong, omitted or added word, or skip) in the sheet.

**Data quality issues** (all listed in `manifest.json` under `sheet_cross_issues`, `filename_repairs`, `cross_issues` and `row_issues`):

- **Six rows point at another row's file.** Six zip WAVs are referenced by no row. In every case the row's position in the sheet names the missing file with the same verse and T/F, so `repair_ikhlas_files` points the row at it:
  - row 421 `ID42V2T`→`ID43V2T`
  - row 1006 `ID249V3F`→`ID248V3F`
  - row 1096 `ID328V3T`→`ID338V3T`
  - row 1208 `ID266V4F`→`ID66V4F`
  - row 1328 `ID286V4F`→`ID186V4F`
  - row 1383 `ID41V4T`→`ID241V4T`

  Only the 1208/1408 pair had conflicting labels, which fits a typo: row 1208's label belongs to `ID66V4F`.
- **Seven pairs of distinct files hold identical audio**, all with matching labels:
  - 118/119 V2
  - 157/166 V4
  - 165/281 V2
  - 165/174 V3 (same samples, different WAV header)
  - 187/193 V1
  - 250/251 V4
  - 76/77 V4

  So `ID<n>` is a submission slot, not a verified person. `dup_of` marks the second copy.
- **Flag contradicts the filename:**
  - Row 26, `ID26V1T`: flag 1 (error) with no error fields. Probably correct.
  - Row 1153, `ID11V4F`: flag 0 (correct), filename F, no fields.
- **Row 892, `ID134V3T`:** flag 0 but `Error explanation ` holds `40`, which is the `Error number` code for "qalqalah of د in يلد". This is probably a shifted cell on an error row.
- **Location wording:**
  - Row 1215 gives `لَمْ` for verse 4, meaning the لم inside وَلَمْ (iẓhār of م).
  - Rows 1209 and 1433 list كُفُوًا twice against two types.
- **Typos:** spelling variants in explanations (`خطا`, `قلقلة الدال`, double spaces) and trailing spaces in `Error type` / `Error location`. These are normalised in the manifest.
- **Unclear audio:** 3 rows say "re-read the verse clearly" (`أعد قراءة الآية بشكل واضح`), so the clip is unclear rather than a specific error.
- **This is the same audio as `azizbekphd/ikhlas_recitations`**, which E1 used. That repo's zip has the same 1,505 files byte-identical (plus macOS `._` resource-fork entries) and lacks only `ID248V3F.wav`, the file the sheet never referenced. So it was built from this sheet and inherited the row-1006 slip. E1 evidence on that set and this set are not independent.
- **Speculation, unchecked:** the "Qur'anic audio websites" part may include professional reciters who are also in our training sources (everyayah, mp3quran). Check before using it as held-out.

**Use in our pipeline.**

- *v1 checker (memorization):* use it as a **false-warning set only**. Its 1,499 unique clips cover 4 verses from 384 submission slots, on mixed consumer and web audio, and the labels report no word-level error. The checker should raise no wrong, omitted or added word, and no skip, on any of them. Exclude the 3 "re-read clearly" clips and the rows flagged above until someone listens. It can't measure detection of memorization errors (there are none), and with one surah it says nothing about tracking.
- *Pronunciation track later:*
  - Short vowels: about 200 ḥarakāt labels, mostly ḍamma on ف in كُفُوًا and kasra/fatḥa on ل, located at word level only.
  - Qalqalah and iẓhār: labelled, but outside the planned classes.
  - No confusable-pair or shadda labels.
  - Clip-level `error` mostly means qalqalah, so a binary error/correct score on this set is a qalqalah detector, not a general one.

## RetaSy Quranic Audio Dataset

**Source.** `RetaSy/quranic_audio_dataset` on Hugging Face, ungated, revision `b1fcc39cbc045f367bb07e39025a0e3aaeabf34f`, 3 parquet shards (1.26 GB). Paper: Salameh et al. 2024, arXiv 2405.02675. Crowdsourced recitations from non-Arab speakers via the Quran Voice app. The README is saved as `README.md` next to the manifest.

**Licence.** None: no `license` in the card metadata and no licence text in the README. Under PLAN.md's data stance (untagged licence: training accepted at our risk), it can be used for evaluation. Training needs an explicit decision.

**Counts.**

| `final_label` | normalised `label` | all rows | Quranic rows | after dedup |
|---|---|---|---|---|
| `correct` | `correct` | 409 | 368 | 362 |
| `in_correct` | `incorrect_pronunciation` | 502 | 440 | 435 |
| `in_complete` | `incomplete_verse` | 25 | 20 | 22 |
| `not_match_aya` | `wrong_verse` | 56 | 42 | 51 |
| `multiple_aya` | `multiple_verses` | 83 | 52 | 79 |
| `not_related_quran` | `unrelated_or_empty` | 153 | 114 | 143 |
| null | `unlabeled` | 5,600 | 4,223 | 5,006 |
| **total** | | **6,828** | **5,259** | **6,098** |

The after-dedup column keeps one row per audio and prefers a labelled copy.

- **Labelled vs README:** 1,228 rows are labelled, against the README's "1166 annotated". 62 are `golden` (expert-labelled).
- **Reciters:** 1,289 (README says 1,287); 210 among the labelled rows.
- **Readings:** `reciter_qiraah` is `hafs` 5,867 times and null 961 times.
- **Judgments per labelled row:** 3 → 958, 4 → 171, 5 → 30, 6 → 4, 0 → 3, null (golden) → 62.
- **Audio:** 8.7 h in total, 1.5 h of it labelled.
- **Texts:** 164 distinct `Aya` texts, 77 among the labelled rows. Al-Fātiḥa is 2,934 rows (817 labelled).

**Label schema.** These are the exact column names, with manifest fields in brackets:

- `audio` (struct `bytes`, `path`; `path` is a UUID `.wav`) → `id` (the UUID stem), `orig`.
- `Surah` (an English name, *including non-Quranic texts*) → `surah`.
- `Aya` (vocalised Arabic text, not a reference) → `aya_text`.
- `duration_ms`, `create_date`, `golden`, `final_label` (→ `label`, `label_raw`).
- `reciter_id`, `reciter_country`, `reciter_gender`, `reciter_age`, `reciter_qiraah`.
- `judgments_num` → `judgments`.
- `annotation_metadata` (a JSON string with `label_N`, `annotatorN_id`, `annotatorN_SCT`, `annotatorN_MCC`, `annotatorN_ACC`, `annotatorN_F1`) → `annotators`: a list of `label`, `annotator`, `mcc`.
- Added: `quranic` (false for prayer texts and the isti'ādha) and `shard`.

The README defines `correct` / `in_correct` as diacritics only, "regardless of the rules of Tajweed".

**Data quality issues.**

- **1,569 rows are not Quran** and must be filtered: Adhan, Iqamah, At-Tahiyyat, Subhanaka, Salawat, Adhkar after prayer, Dua Qunoot and Dua for Protection, plus isti'ādha clips filed under Al-Faatihah. The README's own example is one of these. "Dua from the Quran" and "Ayat Ramadan" are kept as Quranic.
- **730 rows duplicate another row's audio exactly,** in 559 groups (479 pairs, the largest has 9 copies). 555 groups come from the same reciter, so they are resubmissions.
  - 20 groups give *conflicting labels to identical audio*. The README's example clip `0058a4f7…` is one of 6 copies labelled 4× `in_correct`, 1× `correct`, 1× unlabelled.
  - 54 groups mix labelled and unlabelled copies.
- **38 non-golden rows** have a `final_label` that isn't the majority of their annotator labels. 3 rows are labelled with 0 judgments and no annotations.
- **282 clips are under 0.5 s** (39 of them labelled). None are digitally silent.
- **Label noise is high by the authors' own numbers:** crowd accuracy 0.77 and inter-rater agreement 0.63.
- `Aya` is text, not `surah:ayah`. The 164 texts need mapping to references before any tracker use.

**Use in our pipeline.**

- *Tracker:* `wrong_verse` (42 Quranic, about 51 unique) is the planned learner-audio tracker slice from decision 20. `multiple_verses` and `incomplete_verse` are smaller tracker and skip-adjacent slices. `unrelated_or_empty` (114 Quranic) is a no-recitation and rejection slice. All of them need reference mapping and a listen, given the label noise.
- *v1 checker:* `correct` (about 360 unique Quranic) is a false-warning set on non-Arab learner audio, with caveats: it ignores tajweed and the labels are noisy. There are no word-level memorization labels, so detection can't be measured.
- *Pronunciation track later:* `incorrect_pronunciation` is clip-level and diacritics-level with no location, so it only supports clip-level scoring. PLAN.md already lists "reviewed RetaSy clips" for pronunciation eval.
- The 5,600 unlabelled clips are untranscribed learner audio, usable for listening or pseudo-labelling.

## QuranMB v2 (IqraEval)

**Source.** `IqraEval/QuranMB.v2` on Hugging Face, ungated, revision `75afc3391e47ca005af286b2518d8a2edf98da5d`.

- `data/test-00000-of-00001.parquet` and `data/train-00000-of-00001.parquet` are **byte-identical** (sha256 `332b3e4b…62b9`, 220,472,046 bytes). Only `test` is in the card config, and only `test` was used.
- Columns: `ID` and `audio` (struct `bytes`, `path` = `<ID>.wav`).

**Ground truth.** `IqraEval/IqraEval_Test_GT` (`labels_test.csv`, revision `cd27009525365f911ddd773c0b3b70cac82f2f0b`, no licence stated), accessible after access was granted on 28 October. Columns `ID`, `Reference_phn` (what the prompt should sound like) and `Annotation_phn` (what was said). All 1,642 rows join to audio; 97 distinct verses. 1,532 items differ from their reference: 2,372 substituted, 944 deleted and 259 inserted phonemes over 53,117 reference phonemes (6.7%). No speaker column; speakers are still inferred from IDs.

So the access request for `yazins` is not approved on HF's side (still pending, or approved for another account). Once it is, `modal run modal/lab_public_sets.py --only quranmb` does the following:

- Saves `gt/labels_test.csv`.
- Joins the rows on `ID` into `items[].gt`.
- If it finds a reference column and an "as said" column, it counts phoneme substitutions, deletions and insertions per item (`phone_edits`).
- Records the columns, verse count and speaker count under `gt`.

The column detection matches on name, because the CSV couldn't be read, so check `gt.columns` after the first successful run.

**Licence.** None in the audio card. The ground-truth licence is unknown (gated).

**Counts (audio only).**

- 1,642 clips with 1,642 unique IDs. Unique PCM: 1,640, because `00005_00056129` = `00005_00056` and `00005_00070950` = `00005_00070`.
- 1.92 h of audio; clips run 1.2–20.3 s (median 4.0 s).
- IDs are `<5-digit group>_<suffix>`. There are 18 groups (`00000`–`00017`) and 98 base prompts (`00000`–`00097`).
  - 1,142 clips use a 5-digit suffix.
  - 500 use a longer suffix that starts with a base prompt. 499 of these have a twin with the same group and prompt.
- *Speculation:* the group is the speaker (18 speakers, `00003` and `00012` partial), the prompt is a verse or sentence (98 verses), and the long-suffix clips are second takes, possibly scripted-error versions of the same prompt. Confirm against the ground truth.

**Use in our pipeline.** Without the ground truth it is unlabelled read speech: useful for listening, nothing measurable. With it, it is the phoneme-level mispronunciation benchmark for the pronunciation track (short vowels, confusable pairs, shadda), after mapping IqraEval's phoneme set to the teacher's. It carries no memorization-error labels as far as the audio card shows.

## What none of these give us

None of the three has word-level memorization errors (wrong, omitted or added words, skips) with locations. v1 checker *detection* still has to be measured on our scripted-error recordings and spliced sessions. These sets only add false-warning measurements (Ikhlas, and RetaSy `correct`) and a future tracker slice (RetaSy `wrong_verse`).
