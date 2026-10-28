# 14 Public error sets onto the lab volume

Status: resolved
Type: task
Blocked by: none

Spec: `../spec.md` decision 20. Download on Modal, not the Mac.

- Al-Ikhlas error detection (Mendeley `sxtmmr6mvk` v1, `Dataset and Sounds.zip`, sha256 `de8042ac8cc137e83fdcdbed375a24bd51f873c94f7f2cca1e39c60b188c0068`). Deduplicate exact audio and list label inconsistencies.
- RetaSy `quranic_audio_dataset`. Note the missing licence.
- QuranMB v2 audio plus `IqraEval/IqraEval_Test_GT` (access approved on the HF account).

Record provenance, licence and counts in `docs/benchmarks/public-error-sets-v1.md`. Not scored yet.

## Comments

Audio for all three sets is on the `tilawa-lab` volume, but QuranMB's ground truth is blocked, so the status is `ready-for-human` rather than `resolved`. Write-up: `docs/benchmarks/public-error-sets-v1.md`. Code: `modal/lab_public_sets.py`. Tests: `tools/test_lab_public_sets.py` (17 pass). Each `/data/public/<dataset>/` has `orig/`, `wav16k/` (16 kHz mono) and `manifest.json`.

- **Ikhlas** (`/data/public/ikhlas/`, CC BY 4.0, sha256 verified):
  - 1,506 rows: 851 error, 655 correct. 1,506 WAVs.
  - 1,500 unique filenames in the sheet. Six rows point at a neighbour's file and are repaired by sheet position to the 6 unreferenced WAVs.
  - 1,499 unique audio after dedup: 7 identical pairs across different `ID<n>` slots.
  - Every error is tajweed (mostly qalqalah of د) or ḥarakāt. There are zero memorization errors, so for v1 it is a false-warning set only.
  - Rows needing a listen: 26 (flag says error, filename says T, no fields), 1153 (flag says correct, filename says F), 892 (flag says correct, explanation holds code `40`), and the 3 "re-read clearly" rows.
  - `azizbekphd/ikhlas_recitations` (used in E1) is the same audio byte for byte, minus `ID248V3F.wav`.
- **RetaSy** (`/data/public/retasy/`, no licence in card or README, revision `b1fcc39`):
  - 6,828 rows. Labels: 5,600 unlabelled, 502 incorrect pronunciation, 409 correct, 153 unrelated or empty, 83 multiple verses, 56 wrong verse, 25 incomplete.
  - 1,228 labelled vs the README's 1,166.
  - 1,569 non-Quranic rows (Adhan, Tashahhud, duas, isti'adha).
  - 730 exact-duplicate rows in 559 groups; 20 groups have conflicting labels on identical audio. 38 final labels aren't the annotator majority.
- **QuranMB v2** (`/data/public/quranmb_v2/`, no licence in card, revision `75afc33`):
  - 1,642 clips, 1,640 unique audio. The train shard is byte-identical to test.
  - IDs suggest 18 speakers × 98 prompts plus 500 second takes (inferred).
  - **Blocker:** `IqraEval/IqraEval_Test_GT` returns `GatedRepoError: 403 ... you are not in the authorized list` for the Modal `huggingface` token, which is account `yazins`. Approval hasn't landed on HF. Once it has, run `modal run modal/lab_public_sets.py --only quranmb` to join the ground truth and count verses, speakers and phoneme edits.
- Nothing is scored and no metric moves, so there is no scoreboard row.
- QuranMB ground truth fetched after access was granted (28 Oct): 1,642/1,642 joined, 97 verses, 3,575 phoneme edits over 53,117 reference phonemes, 1,532 items with errors. IqraEval's phoneme set still needs mapping to the teacher's before any scoring.
