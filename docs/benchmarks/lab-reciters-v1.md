# Lab held-out reciters v1

Lab ticket 01. Lists the mp3quran Hafs reciters with ayah timings that are in no training data of the teacher or the student, and in none of the planned training sources, then splits them into practice and exam by reciter (spec decisions 4 and 8). Machine-readable: `lab-reciters-v1.json`. Regenerate:

    uv run --python cpython-3.12-macos-aarch64-none tools/lab_reciters.py [--refresh]

Only JSON metadata is fetched (plus one HEAD request per held-out reciter for the surah 67 mp3 size). No audio is downloaded.

## Result

**32 held-out reciters, 16 practice and 16 exam.**

| | reads |
|---|---|
| mp3quran reads with ayah timings | 115 |
| Hafs moshaf (name contains حفص) | 103 |
| reciter is in the teacher's training data | 19 |
| reciter is in the student's training data | 0 (tlog has no names, see caveats) |
| reciter is in a planned source (QUD, zaibihassan, everyayah) | 70 |
| uncertain match only | 0 |
| mujawwad or "featured recitation" moshaf | 6 (all also leak) |
| hadr-like speed | 4 (3 also leak) |
| **held out** | **32** |

A read can be excluded for more than one reason. Leakage alone removes 70 of the 103; style, speed and timing remove one more (Abdulmohsin Al-Obaikan, read 69, Al-Mulk at 0.72 of the median duration).

| split | reciters | mean speed (vs median) | mean kbps | under 110 kbps | timed surahs |
|---|---|---|---|---|---|
| practice | 16 | 1.015 | 128 | 4 | 1,824 |
| exam | 16 | 1.036 | 142 | 4 | 1,823 |

Practice: Khalid Abdulkafi, Dawood Hamza, Shirazad Taher, Saleh Alsahood, Saleh Al-Habdan, Abdulhadi Kanakeri, Abdulwali Al-Arkani, Neamah Al-Hassan, Mohammad Saleh Alim Shah, Jamaan Alosaimi, Mohammad Rashad Alshareef, Sayed Ahmad Hashemi, Khalid Mohammadi, Issa Omar Sanankoua, Anas Alemadi, Hassan Aldaghriri.

Exam: Ibrahim Al-Asiri, Ahmad Al-Hawashi, Ahmad Saber, Zaki Daghistani, Salah Alhashim, Adel Ryyan, Abdulbari Mohammad, Abdulmohsin Al-Harthy, Abdulrahman Alsuwayid, Abdulrahman Aloosi, Mohammad Khalil Al-Qari, Abdulrahman Al-Majed, Abdullah Al-Khalaf, Mohammad Albukheet, Hasan Saleh, Saleh Alshamrani.

## What the models were actually trained on

- **Student:** tlog-v2 only (`modal/student_prep.py` reads `tarteel-ai/tlog` shards; `tools/student_manifest.py` builds from tlog-v2 buckets; `student-v1.md` and `student-v2.md` say "Training used tlog only"). The `full` tier starts from English LibriSpeech Zipformer-M weights. Augmentation adds `obadx/freesound-commercial-50k-noise-only` and `EaseZh/rirs_noises`, which contain no recitation. QUD only goes through teacher passes for evaluation (`modal/qud_pass.py`), not training.
- **Teacher:** `obadx/muaalem-model-v3_2` lists only `obadx/muaalem-annotated-v3` as its dataset; its base is `facebook/w2v-bert-2.0`. The dataset's `reciters_metadata` config has 22 reciters across 27 moshafs, mostly Egyptian murattal (Husary, Minshawi, Abdul Basit, Banna, Mustafa Ismail and others).
- **Planned sources** (PLAN.md Data table; the ticket says to exclude them): QUD `mushafs` catalog (104 mushafs from all riwayat, matched at reciter level), the zaibihassan top-level folders, and the everyayah.com `recitations.js` index. That index is a superset of `tarteel-ai/everyayah` and its HF mirrors. `obadx/qdat` is anonymous learner audio and can't be matched.

## Matching

Names are compared in Arabic and in English against every source name.

- **Arabic:** diacritics and tatweel are removed, and alef, ta marbuta and ya forms are unified. `عبد`/`ابو` are joined to the next word, honorifics and `بن` are dropped, and the `ال` prefix is stripped.
- **English:** labels in parentheses, articles (`al`, `ash`, `ar`…), `bin`, `ibn` and honorifics are dropped. Each word is reduced to a consonant skeleton: digraphs folded, d/t merged, vowels and medial y/w removed, double letters collapsed. This lets "Saood bin Ibraaheem Ash-Shuraym" match "Saud Al-Shuraim".
- **Match rule:** first and last tokens both match, with normalised Levenshtein ≥ 0.75. Tokens of 4 characters or fewer must match exactly, so محمد ≠ محمود. A one-token source name such as everyayah's "Husary" or "Ghamadi" that equals the reciter's surname counts as an **uncertain** match. Uncertain-only reads would be excluded, but all 14 surname-only hits (8 reads) are on reciters that also match firmly.
- **Manual decision:** read 123 (Mishary Alafasy) against the teacher. muaalem-annotated-v3 spells him "مشاري راشد الفاسي / Mishary Rashid Al-Fasi". His moshaf 19.0 is islamway collection 213, Alafasy's Cairo murattal, so he is excluded as teacher leakage.
- **Rejected rule:** an earlier token-subset rule matched Mohammad Khalil Al-Qari to Mahmoud Khalil Al-Husari, and two other pairs wrongly. It was dropped.

Every exclusion lists its matched source names and the method in `excluded[].evidence`. Each held-out reciter carries its `nearestSourceName`. All 32 were also checked by eye against the full source lists.

## Split

Reciters are paired within two bitrate bands: under 110 kbps on the surah 67 mp3, and the rest. Within a band they are sorted by speed, and each neighbouring pair is split by a seeded coin flip (seed `20261028`). Speed is the Al-Mulk recited span (ayah 1 start to ayah 30 end) divided by the median over all Hafs murattal reads (380 s).

## Caveats

- **tlog leakage is unverifiable.** It is crowd-sourced with no reciter names, and users may have recorded a professional recitation playing. This is the student's only training audio.
- **w2v-bert-2.0 was pretrained** self-supervised on about 4.5M hours of unlabelled multilingual audio. Any reciter could be in it. No labels are involved.
- **Name matching only.** Two reciters with the same name, or one reciter under very different spellings, could be misplaced.
- **Same voice, other riwayah.** Two reciters are excluded only through another riwayah: Ahmed Al-Trabulsi via QUD Qalon and Ibrahim Al-Dosari via everyayah Warsh. Both names match closely, so they were left excluded.
- **Speed and quality are proxies.** They come from one surah's timings and the mp3 bitrate, not from listening. Ahmad Saber's Al-Mulk file is 65 kbps, the lowest in the set (exam). Zaki Daghistani (1.40) and Abdulbari Mohammad (1.35) are the slowest.
- **Timings are only checked on Al-Mulk.** The check is 30 ayahs in monotonic order. Timing quality on other surahs is a later ticket's job.
- **Future training must blocklist these reciters.** The held-out set is mostly Gulf imams, while the teacher's data is mostly Egyptian. Any new training source (for example mp3quran itself) must drop these 32 reciters, or the exam leaks.
