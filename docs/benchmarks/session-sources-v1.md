# Clean continuous-session sources — 12 October 2026

Use existing QUD long recordings for the first baseline. These are professional diagnostic sessions, not independent learner-phone acceptance evidence. Future user mistake recordings remain a separate pending input. After research, three originals were acquired for the implementation baseline.

## Verified local material

`.scratch/bench/qud/` contains 16 recordings, 356.75 minutes, 12 named reciters, ordered word truth and teacher evidence. `modal/qud_pass.py` identifies the upstream dataset as [QUD-Technologies/quran-alignment-benchmark](https://huggingface.co/datasets/QUD-Technologies/quran-alignment-benchmark), config `v1`; it reads embedded audio from parquet and runs the teacher in 30-second windows. Existing evidence is from continuous source audio, not concatenated per-ayah decodes. It is still windowed teacher inference, not the target student streaming listener. Acquisition verified revision `76c3d804d082c3cc189601faf13a41996e17b6dc`: the owner card declares Hafs, reviewed word truth, and CC BY 4.0 for annotations/descriptive columns/repository only. Audio is explicitly excluded from that license. No commercial audio clearance is inferred.

Useful clean-single-passage candidates from local metadata:

| Recording | Minutes | Reciter | Conditions |
|---|---:|---|---|
| afasy-baqarah | 79.95 | Mishary Al-Afasy | hadr, no noise flag, 28 repeats |
| husary-fath | 24.84 | Mahmoud Khalil Al-Husary | mujawwad, no noise flag, 7 repeats |
| abdulbasit-raad | 38.08 | Abdul Basit Abdul Samad | mujawwad, no noise flag, 3 repeats |
| diban-sad | 4.38 | Ahmed Diban | hadr, no noise flag, 1 repeat |
| jaber-nisa | 0.97 | Ali Jaber | hadr, no noise flag, no repeats |

These durations describe entire recordings, not clean-speech minutes. Truth reconciliation must preserve repeats and classify introductory/surrounding non-Quran speech. Prayer tracks and multi-surah tracks need explicit transition/non-Quran annotations. Metadata names do not independently prove a single voice (especially muallim/children tracks).

## Additional sources verified with owners

**QUD universal ayahs.** Its [dataset card and preview](https://huggingface.co/datasets/QUD-Technologies/quranic-universal-ayahs) expose actual consecutive Baqarah ayahs 1–24 under one source URL and increasing source offsets. Catalog fields include stable reciter ID, Hafs/other riwayah, recording context, and by-surah versus by-ayah audio. This is concrete consecutive coverage, but the default preview's riwayah must still be joined to the catalog. Prefer the original chapter audio with offsets; otherwise retain stitching provenance. Word timing preserves repeats and is alignment-generated/community-verified, not independent manual truth. Metadata is CC-BY-4.0; the card explicitly leaves audio rights with upstream owners. No general commercial audio clearance follows from that tag. Current viewer total is 565,462 rows, differing from the older note; don't use totals as eligible Hafs hours.

**Muaalem annotated.** The [owner card](https://huggingface.co/datasets/obadx/muaalem-annotated-v3) declares MIT, Hafs convention metadata and 847.99 hours/286,537 audio segments. It has reciter/moshaf IDs, range start/end, segment indexes and timing, with 16 kHz mono audio. Those fields enable continuity audits but do not establish complete consecutive word coverage or reconstruct original breath gaps automatically. Viewer currently fails for the default split, so no consecutive rows were verified here. Teacher Muaalem itself was trained on this corpus: using it for teacher evidence is an in-domain diagnostic, not independent listener evaluation. Prefer QUD first; importing this source adds reconstruction and leakage checks.

**Zaibihassan full-surah recordings.** The [owner card](https://huggingface.co/datasets/zaibihassan/Quranic-Recitation-Data/raw/main/README.md) provides full-surah 16 kHz mono Opus and millisecond word-index protobuf timing. It declares Apache-2.0 and commercial permission. Its 135 catalog entries include Warsh, other traditions, duplicate aliases, children and combined-reciter collections; that is not 135 independent Hafs speakers. Suitable single-reciter candidates exist, but explicit Hafs evidence and actual file/timing validation remain necessary. Claimed millisecond accuracy is unvalidated. Neither advertised full-surah packaging nor timing proves source was genuinely continuous rather than pre-stitched. No audio downloaded or row-count claim independently confirmed.

## Leakage and baseline procedure

Local `tools/student_manifest.py`/`modal/student_prep.py` describe tlog training; speaker identity there is approximate clusters, so professional overlap cannot be excluded by file name. `tools/e1b_build.py` uses EveryAyah Husary and Minshawy Juz 29–30; hold all donors/recipients and recording aliases together when splitting. Current student checkpoint provenance must be audited before declaring any source held out. Husary Fath is outside those Juz ranges but is the same person; passage separation alone does not satisfy reciter-held-out evaluation. The older [data research note](../../.scratch/v1-roadmap/research-data-2026-10-07.md) already requires person/recording deduplication across these corpora.

Start with a pinned QUD recording, verified intended start context, both reading and memorization modes, continuous audio through the streaming listener, and independently audited word occurrences/non-Quran spans. Record checkpoint training overlap as unknown until established. Existing teacher replay can validate scorer plumbing, but must be labelled as such. Then expand separately by studio, prayer and noisy professional conditions; do not pool away condition-specific failures. Local original WAVs are now acquired for Jaber Inshiqaq, Jaber Nisa and Diban Saad from the pinned embedded upstream parquet. Fresh continuous student decodes and provenance hashes are saved. The remaining recordings have not been acquired; keep audio outside git and report real gaps rather than padding missing ayahs with another reciter.
