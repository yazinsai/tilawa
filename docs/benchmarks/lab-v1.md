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
| `sessions/<id>.<listener>.<studio\|phone>.evidence.json` | listener evidence, written by ticket 05 |
| `manifest-v1.json` | split, slices, audio and label paths, inline session; `sessions-cli --manifest` reads it |
| `plan-v1.json` | sample-exact pieces and phone parameters per session |
| `summary-v1.json` | `lab-v1.json` plus the reciter list |
| `review/v1-check.json` | the ticket 07 listening check (20 splices, 20 cases) |
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

## Known limits

- **Linked seams:** 307 of 649 seams cut inside continuous speech, at linked boundaries 3 dB or more below the ayah level, not at pauses. The phoneme rules keep the tail and head forms consistent, but coarticulation across a link is not modelled. The v1 listening check covers these.
- **Few mutashabihat swaps:** there are only 15. Twin ayahs that differ in exactly one word are rare, and refrain words are mostly said without a usable gap.
- **Strict slice:** the slice has 6 sessions, too few for more than a smoke number. One strict start is in An-Nisa (4:122) rather than a refrain surah.
- **Restarts:** a restart goes back to the first word of the current ayah after a pause. It does not model a mid-word break.
- **Planted words are real words:** wrong and added words are whole words said correctly by the same reader. Pronunciation slips and mid-word hesitations are not planted.
- **Source precision:** word timings come from teacher alignment (`lab-align-v2`), so label precision is limited by it. Words either side of a seam score at least 0.5, and donor words at least 0.7.
