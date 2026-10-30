# 02 Fetch mp3quran surahs and align words

Status: resolved
Type: task
Blocked by: 01

Spec: `../spec.md` decisions 4 and 15.

A Modal job downloads the full-surah MP3s and ayah timings for the held-out reciters onto a lab volume, converts them to 16 kHz mono, and uses the teacher's CTC to force-align words inside each timed ayah.

## Acceptance

- Each surah file on the volume has ayah and word timings, plus the source URL, hash and API revision.
- Each ayah is checked against the alignment; any mismatch over 300 ms is listed and excluded.
- Nothing is downloaded or decoded on the Mac.

## Comments

28 Oct 2026. Done in `modal/lab_fetch_align.py`, with the logic in `tools/lab_sources.py` and tests in `tools/test_lab_sources.py`. Full write-up and output schemas: `docs/benchmarks/lab-sources-v1.md`.

- **Pool.** 20 surahs per reciter:
  - always 55, 77, 54 and 26;
  - then 8 host surahs (160–450 words), 5 medium (450–1,500) and 3 long (over 1,500), each a seeded pick from the least-covered juz.

  That makes 63 distinct surahs.
- **Data.** 640 surahs, 192.6 h, 48,946 ayahs, 552,556 aligned words, on volume `tilawa-lab` under `data/sources/mp3quran/<readId>/<sss>.{wav,source.json,evidence.npz,align.json}`, plus `catalog-v1.json` and `report-v1.json`. Every surah file has the source URL, sha256 and byte count, and the raw timing JSON with its sha256 and fetch time (mp3quran has no API revision). Nothing was downloaded or decoded on the Mac.
- **Alignment.**
  - Teacher CTC Viterbi (`e1_report.viterbi`) on ref-v2 phonemes. Each ayah is aligned together with its neighbours' text, runs of one phoneme are collapsed to one token, and a garbage-tolerant blank absorbs stray speech. Ayah 1 is also tried with a basmala prefix.
  - In the pilot, decodes were read by hand and dB profiles checked at 50 ms around boundaries. Aligned ayah ends and starts fall on the energy drop and onset to within about 50 ms. The full-set medians are −20 ms (onset) and −35 ms (offset).
  - 96.5% of ayah-boundary pauses lie inside the aligned gap. Word boundaries are 3.5 dB quieter than the same search at word middles.
  - Re-alignment is byte-identical.
- **Consistency.** 6,811 ayahs are excluded (13.9%) and listed with their reasons in `report-v1.json`:
  - 2,127 are in 28 surahs whose timing doesn't fit the audio. Read 135 (exam) is fully excluded for timing drift, and 299 has four truncated timings.
  - Of the 5,436 PER exclusions, 4,244 decode at least 10% longer than their text, which means real repeats or restarts.
  - About 130 are muqatta'at ayah 1s.

  That leaves 149.8 h, in 933 clean runs of 3 minutes or more (practice 488, exam 445).
- **For ticket 03.**
  - Cut at `words[].cutAfterS` and prefer a low `cutDb`; murattal links most words.
  - Filter by `ayahs[].excluded`.
  - The exam set has 15 usable reciters.
  - `basmala` is set where it was found (470 surahs).
- **Cost and time** (estimate). 2.9 L4-hours of teacher time, about 45 min to fetch on 32 containers, and about 12 min per alignment pass. Roughly $10–20 in total.
- **Scoreboard.** No row was added: this is source data, and no metric is computed from it yet (spec decision 1 retires the scoreboard).
