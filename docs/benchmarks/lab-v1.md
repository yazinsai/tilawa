# Lab test set v1

40 spliced sessions of real mp3quran recitation with planted mistakes, in a studio and a phone copy (`.scratch/lab/issues/03-session-builder.md`). Counts are in `lab-v1.json` beside this file.

- Build: `modal run modal/lab_build_sessions.py` (`--pilot` renders 2 sessions; `--force` rebuilds). Logic: `tools/lab_sessions.py`.
- Test: `uv run --python cpython-3.12-macos-aarch64-none --with numpy --with scipy --with pytest pytest tools/test_lab_sessions.py`
- Core validation: `python3 tools/lab.py run --cmd 'pnpm --filter @tilawa/core exec tsx src/bench/lab-validate-cli.ts'` (every session JSON and the manifest's inline copies through `validateSession`).
- Seed 20261028. The same seed and inputs give byte-identical files: a forced rebuild reproduced every WAV hash and wrote no changed JSON.

## Files

On the `tilawa-lab` volume under `/lab/data/lab-v1/`:

| Path | What |
| --- | --- |
| `sessions/<id>.json` | BenchSession plus `said`, `splices`, `meta` |
| `sessions/<id>.studio.wav`, `.phone.wav` | 16 kHz mono PCM16 |
| `sessions/<id>.checks.json` | render checks (below) |
| `sessions/<id>.<listener>.<studio\|phone>.evidence.json` | listener evidence (`spliced-clean-int8`, `teacher`), written by ticket 05 |
| `decode-v1.json` | per listener: model hashes, flags, source commit, run id, per-file hash, frames and seconds |
| `checks-v1.json` | the automated checks (ticket 04): per-case pass/fail with reasons; `pnpm --filter @tilawa/core lab` reads it |
| `manifest-v1.json` | split, slices, audio and label paths, inline session; `sessions-cli --manifest` reads it |
| `plan-v1.json` | sample-exact pieces and phone parameters per session |
| `summary-v1.json` | `lab-v1.json` plus the reciter list |
| `review/v1-check.json` | the ticket 07 listening check (20 splices, 20 cases), with every splice that failed the automated checks |
| `review/v1-check.base.json` | the builder's queue before the checks swapped failed splices in |
| `aug/noise.npz`, `aug/rir.npz`, `aug/bank-v1.json` | the phone-copy noise and RIR bank with provenance |

Ids are `lab-v1-p01`–`p20` (practice) and `lab-v1-e01`–`e20` (exam), ordered by reader, surah and start.

## Counts

| | practice | exam | all |
| --- | --- | --- | --- |
| sessions | 20 | 20 | 40 |
| reciters | 16 | 15 | 31 |
| minutes | 92.1 | 94.7 | 186.8 |
| ayah-1 starts | 4 | 4 | 8 |
| strict starts | 3 | 3 | 6 |
| clean sessions | 5 | 5 | 10 |
| planted mistakes | 182 | 188 | 370 |
| wrong word (plain) | 55 | 59 | 114 |
| mutashabihat swap | 7 | 8 | 15 |
| omitted word | 35 | 43 | 78 |
| added word | 53 | 57 | 110 |
| skipped passage | 32 | 21 | 53 |
| repeats / restarts (not mistakes) | 8 / 12 | 8 / 12 | 16 / 24 |
| events with `forms_other_ayah` | 16 | 14 | 30 |

Sessions run 182–415 s. Every session has one repeat or restart. Planted mistakes come at 0.37 per ayah in the 30 sessions with mistakes, so about one per 2.7 ayahs. Mutashabihat swaps are labelled `wrong_word` with `forms_other_ayah: true`; `splices[].mutashabihat` and `meta.mutashabihat` mark the slice. They fall in 6 sessions, mostly in Ash-Shu'ara (26). Surah 26 has 7 sessions; every other surah has 1 or 2 (31 surahs).

Exam excludes read 135, whose timings drift against its audio in every surah (`lab-sources-v1.md`).

## Sessions

Each session is one window of one catalog recording over consecutive included ayahs. It starts in the pause before an ayah, or just before the basmala at ayah 1, and ends in the pause after an ayah. Excluded ayahs never fall inside a window. A reader gets at most one session per surah.

- **Normal starts:** the disambiguation point (`disambiguation-v1.json`) is within 5 words of the first word.
- **Strict starts:** 6 or more words are needed, inside long shared runs. They land in 26 (3), 54 (2) and 4 (1). The Al-Rahman and Al-Mursalat refrains are disambiguated within 4 words, so they count as normal starts.
- **Ayah-1 starts:** the window begins at ayah 1.
- **Mutashabihat focus:** 3 per split; these take the window with the most clean twin swaps for that reader.

Labels have track `recitation` and context `none`. They use `source: "stitched"`, with one segment per piece (output and source times, gain).

Truth words are timed in the output: a planted word appears as said, and an omitted word is absent. Locations follow the checker:
- added word: the word after the insertion;
- omitted word: the omitted word;
- skipped passage: the first word plus `through`.

Omission and skip intervals run from the end of the previous word to the start of the next. A repeat or restart opens its own `repeat` or `restart` span, holding only the re-said words, and each re-said word is a `nonErrors` entry. A basmala present at an ayah-1 start becomes a `non_quran` span. All events are `confirmed`.

## How planting is constrained

Edits are cut only at word boundaries, which are classified from the ref-v2 phonemes. Each word's slice of the continuous ayah phonemes (`p`/`o`) is compared with its pause form (`iso`). A boundary is usable when it meets all of these:

- **Kept tail:** the word before it keeps its ending: the pause form plus a case vowel, a long vowel, a closed consonant, or a tanween said in full before a throat letter.
- **Kept head:** the word after it starts as in isolation, or drops only the connecting hamza.
- **Cut in the gap:** the aligned cut lies within 20 ms of the gap between the two words.
- **Quiet enough:** the boundary dips at least 3 dB below the ayah level, or 12 dB at an ayah end.
- **Confident alignment:** both words have teacher scores of at least 0.5.

Words that are empty in the phonemes, shortened madd, nasalised endings, and assimilated or merged endings are refused.

A seam joins tail A to head B only when B has the head kind A was said before. A pause joins a pause or a firm head. A long vowel never meets a hamza or its own letter. A closed `n` meets only throat letters. Other closed letters never meet a letter they assimilate into (for example م before ب, ل before ر).

- **Wrong and added words** come from the same reader, from a non-excluded ayah with a word score of at least 0.7. Both of the donor's boundaries must join the host's. Donors come from the same recording first, outside the session window or at least 30 s from the plant; 234 of 239 donors do. A wrong word differs in letters from its neighbours and has a similar phoneme length. An added word has 3–10 phonemes.
- **Mutashabihat swaps** replace a word with the same word position of a twin ayah, one that has the same word count and differs in only that word. The swap must make `forms_other_ayah` (the `tools/help_import.py` rule) true. Ayahs with the same text as the prompt (refrains) are not counted as other ayahs.
- **Omitted words** have at least 3 phonemes and differ from both neighbours.
- **Skipped passages** are one whole ayah of 3–30 words, at most 25 s long, with pauses on both sides and text unlike its neighbours.
- **Placement:** mistakes start after the disambiguation point and at least 20 s in, and stop two ayahs before the end. They are at least 8 words and 6 s from any other edit, and the class pass skips ayahs that already hold an edit. Mutashabihat swaps are placed first, then a weighted class per ayah, one every two ayahs, among classes with a clean site there. The quietest of the best three sites is preferred.
- **Rendering:** each seam has a 10 ms equal-gain raised-cosine crossfade centred on the cut, so output length is the sum of piece lengths. Sessions fade in and out over 20 ms. A donor piece is gain-matched to the host ayah level, limited to ±6 dB.

## Phone copy

The studio render is convolved with a room impulse response, trimmed to its direct-path peak so there is no delay and energy-normalised. It is then band-limited to 120 Hz–7 kHz (zero-phase) and mixed with noise at 8–22 dB SNR (median 15.5) against the active speech level. Noise clips are looped in seeded order with 10 ms fades. The result is peak-limited to 0.89. Each session's RIR, clips and SNR are fixed by the seed and recorded in `meta.phone`.

The bank uses only sources the student never saw:

- **Noise:** 400 clips of 4–12 s from `obadx/freesound-commercial-50k-noise-only` shards 100–103. The student uses shards 0–2.
- **Room impulse responses:** 200 simulated small-room and medium-room RIRs from OpenSLR-28 (`EaseZh/rirs_noises`). The 14 that matched the student's bank exactly were dropped.

## Render checks

These checks are in `<id>.checks.json` for every session.

- **Lengths:** output length equals the plan and the label `durationS` to the sample.
- **Piece copy:** away from fades, each piece equals its source times its gain (max error 1.5e-5, PCM16 rounding).
- **Seam levels:** the 20 ms either side of each of 649 seams matches the same source audio within 0.17 dB.
- **Clicks:** the largest sample step inside a crossfade is at most 1.9 times the 99th percentile of steps around it.
- **Seam depth:** seams sit at a median of 17 dB below the session's active speech level (10th to 90th percentile: −28 to −10 dB; loudest −5 dB, at linked boundaries).
- **Words at their labels:** every said word was found at its labelled time. The 60 ms around the middle of its source audio correlates above 0.999 at zero lag with the output, for all 9541 checkable words. The other 154 words are shorter in the output because a cut falls inside their aligned span, so they were not checked.
- **Phone lag:** the cross-correlation lag between the phone and studio copies has a median of 0. Some windows read up to 25 ms in reverberant, low-SNR copies. The delay is zero by construction (RIR trimmed at its peak, zero-phase filter), so these readings are a limit of the measurement.

## Decode

`modal run modal/lab_decode.py --listener <name> --model <dir under /lab/data>` decodes every session and variant with a student checkpoint; `--listener teacher` runs the teacher. Options: `--only <ids>`, `--variants`, `--push-ms`, `--force`. It refuses to mix evidence of a different model under one listener name without `--force`. Files are skipped when present, so a rerun only fills gaps.

| | `spliced-clean-int8` | `teacher` |
| --- | --- | --- |
| model | `models/spliced-clean-int8`: `model.onnx` `714e98ad…af1b`, `manifest.json` `785f5eb4…442f` | `obadx/muaalem-model-v3_2` @ `01a1ef9f` |
| how | SDK `decode-cli` (streaming, free recitation), onnxruntime-node CPU 1 thread, 20 ms pushes, top-k posteriors at the model's 40 ms | 30 s windows every 25 s, frames from the nearer centre, bf16, top-5 posteriors, greedy phonemes |
| compute | one 2-CPU container per file, 3148 container-s, 21–68 s per file | L4, 437 GPU-s, 3–10 s per file (RTF 0.02) |
| wall | 228 s | 600 s (Modal ran one L4) |
| run | `decode-spliced-clean-int8-d407c89-20261028T123012Z` | in-app |

80 files each (40 sessions × studio, phone; 22,418 s of audio). Every file's frame count matches its label `durationS` within 0.1 s. Cost was under $1.

## Automated checks

`modal run modal/lab_check.py` (one CPU container per session × variant) runs the checks on teacher evidence and writes `checks-v1.json` and the review queue. Logic: `tools/lab_checks.py`. Test: `uv run --python cpython-3.12-macos-aarch64-none --with numpy --with rapidfuzz==3.14.1 --with quran-transcript==0.6.4 --with pyarrow --with pytest pytest tools/test_lab_checks.py`. Studio is the acceptance copy; phone is reported too.

- **Planted edits:** the window runs from the word before the edit to the word after, 0.1 s padded, placed by a forced alignment of the whole session's said phonemes (CTC Viterbi with a garbage blank). Two hypotheses are scored over it: what was said (neighbour + planted + neighbour) and the original text. A pass needs all of the following:
  - the said version wins by at least 3.0 nats;
  - the teacher's greedy decode is closer to the said version than to the original;
  - local PER against the said version is at most 0.25.

  For an omission or skip, the original holds the missing words. For an addition, repeat or restart, it lacks them.
- **Clean stretches:** runs of unedited words inside one source piece, never across a seam. The check is the canon PER of the teacher's greedy decode against the said text. The bar is 2 × the gold-set teacher PER (2 × 3.39% = 6.78%). Stretches under 30 phonemes may have up to 2 edits instead.
- **Timings:** each ayah occurrence's first-word start and last-word end, label vs alignment, within 0.3 s.
- **Reciters:** the reader is `heldOut`, not excluded, and on the session's split in `lab-reciters-v1.json`.

| studio (phone) | n | pass |
| --- | --- | --- |
| wrong word | 114 | 112 (102) |
| mutashabihat | 15 | 14 (13) |
| omitted word | 78 | 74 (71) |
| added word | 110 | 104 (107) |
| skipped passage | 53 | 53 (52) |
| repeat / restart | 16 / 24 | 16 / 24 (15 / 24) |
| clean stretches | 450 | 435 (398) |
| sessions with timings in 0.3 s | 40 | 32 (23) |
| reciters | 40 | 40 |

Studio clean PER over the 62,660 checked phonemes is 2.15% (phone 3.99%). Phone failures are mostly teacher errors on degraded audio, so they don't count against the data.

**Failures (studio).** These are logged in `checks-v1.json` (`failed`, `failedClean`, `failedTiming`) with reasons, and none is patched.

- **13 edits.** 6 fail only on local PER 0.26–0.36: e07-x5, e08-x21, e12-x3, e12-x4, e19-x20 and p13-x15. The teacher hears the planted words but blurs the seam. The other 7 are:
  - e18-x7 and e18-x10: the added donor حين (51:43:8) loses its ح at a link head, and the teacher prefers the original (LLR −6.1, −7.7).
  - p07-x5, p07-x7 and e14-x5: donors labelled in continuous form but said in pause form. Example: قُوَّةً is labelled قُووَتَن but said قُووَه.
  - e11-x9: an unlabelled natural repeat of فخر عليهم in the host (16:26) before the added word gives local PER 0.64.
  - e17-x19: the greedy decode ties between said and original, with local PER 0.31.

  Edits with any linked seam failed 12 of 339; edits with only pause seams failed 1 of 71.
- **15 clean stretches**, at PER 7–17%. 12 are dominated by insertions whose inserted chunks repeat nearby words. These are natural repeats inside source ayahs: the source alignment's garbage blank absorbed them, so the labels lack them. The rest are reciters the teacher finds hard (session PER: p09 8.7%, e13 8.8%, e03 7.3%).
- **10 timing occurrences in 8 sessions.** Most are an ayah-final word ending 0.45–0.56 s off before a pause, repeat or restart, with its start in place. That is a limit of measuring fading madd tails. Two are larger:
  - p10 (26:46, +2.9 s end): the alignment is pulled by a teacher spike in a 3 s pause.
  - e05 (54:35, start 0.8 s): an idgham link across the ayah boundary.

  None is a label error on inspection of the evidence.

**What uses the failures.**

- **Scorer:** `pnpm --filter @tilawa/core lab` reads `checks-v1.json` beside the manifest (`--checks <path|none>`). It adds `dataCheck: {pass: false, check, reasons}` to the matching cases without changing any count:
  - a planted mistake that failed;
  - a false warning within 1 s of a failed edit or clean stretch.
- **Review:** every failed splice is in `review/v1-check.json`. Each one replaced the last passing splice of the same kind and class: 13 replaced, ids otherwise unchanged, queue revalidated. Failed items carry `check: {pass: false, reasons}`. Rerunning the build's plan stage rewrites the queue, so rerun the checks after it.

**Builder changes these point to.** These are not applied in v1, which stays as built until ticket 07's listening check.

1. **Pause-form donors.** A wrong or added donor followed by a pause in its source is labelled with its continuous-form tail. This affects 16 of 239 donors:
   - e09-x1, e09-x2, e14-x5, e16-x3, e18-x14;
   - p03-x2, p03-x3, p03-x8, p05-x8, p05-x11, p07-x5, p07-x7, p12-x2, p14-x8, p16-x14, p19-x16.

   When the donor's source boundary is a pause, `said` should use its pause (`iso`) tail.
2. **Natural repeats in host ayahs.** Ayah inclusion (squeezed PER ≤ 0.15) lets through ayahs where the reciter repeats a phrase. Two fixes:
   - refuse ayahs whose teacher decode is more than about 5% longer than the text, or whose garbage-blank frames hold confident non-blank phonemes;
   - or label the repeat as a `repeat` span.
3. **Linked heads with weak onsets.** Refuse a donor whose head is a fricative or pharyngeal at a linked seam, like the ح of حين; or require a pause seam on the donor's head side.

## Known limits

- **Linked seams:** 307 of 649 seams cut inside continuous speech, at linked boundaries 3 dB or more below the ayah level, not at pauses. The phoneme rules keep the tail and head forms consistent, but coarticulation across a link is not modelled. The v1 listening check covers these.
- **Few mutashabihat swaps:** there are only 15. Twin ayahs that differ in exactly one word are rare, and refrain words are mostly said without a usable gap.
- **Strict slice:** the slice has 6 sessions, too few for more than a smoke number. One strict start is in An-Nisa (4:122) rather than a refrain surah.
- **Restarts:** a restart goes back to the first word of the current ayah after a pause. It does not model a mid-word break.
- **Planted words are real words:** wrong and added words are whole words said correctly by the same reader. Pronunciation slips and mid-word hesitations are not planted.
- **Source precision:** word timings come from teacher alignment (`lab-align-v2`), so label precision is limited by it. Words either side of a seam score at least 0.5, and donor words at least 0.7.
