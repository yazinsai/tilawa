# 10: Streaming benchmark and performance gate

**What to build:** A benchmark harness reports how the full streaming tracker performs against the v1 acceptance bar on the QUD alignment benchmark and `nurlingo/quran-recitation-detect-split`, plus a performance budget check.

- **Location metrics:**
  - correct-ayah-within-5-seconds rate;
  - false-lock rate;
  - unresolved rate (never locked);
  - false-jump rate;
  - shortlist recall;
  - time to first lock.
- **Shared openings:** for passages whose opening is genuinely shared, lock delay is also measured from the point where the audio becomes distinguishable.
- **Configurations:** run phoneme-only, plus posterior reranking if the release configuration includes it.
- **Performance:** a fixed replay records processing time per second of audio against the real-time budget for the device floor.

**Blocked by:** 09; also 05 if the release configuration includes reranking.

**Status:** done

- [x] The benchmark runs end to end on both datasets and writes a versioned report.
- [x] False locks and unresolved cases are reported alongside the 5-second rate.
- [x] The distinguishable-point lock delay is reported for shared-opening cases.
- [x] The performance gate fails when replay falls below real time at the budgeted speed factor.
- [x] The report states which configuration (with or without reranking) was measured.

## Comments

- **QUD:** done end to end.
  - `modal/qud_pass.py` decodes the 16 recordings (357 min) with Muaalem in 30 s windows.
  - `tools/export_qud.py` writes the evidence to `.scratch/bench/qud/`. That folder is gitignored, about 35 MB.
  - `pnpm bench [--posteriors]` writes versioned reports: `docs/benchmarks/qud-v1-{decoded,posteriors}.{json,md}`.
  - Scoring:
    - Passage starts are discontinuities of more than 2 ayahs in Quran order.
    - A lock is right if it names an ayah recited in the last 4 s. During a basmala or isti'adha, the next recited ayah also counts.
    - A false lock is a new wrong location shown before the right one. Still showing the previous passage is stale, not false.
- **nurlingo:** not run. It's ASR text without diacritics, with no audio and no phonemes. The phoneme tracker can't take it as input. Options: drop it from this ticket, or phonemize its consonant skeleton (a separate, lossy matcher). Needs a decision, hence ready-for-human.
- **Results (decoded; reranking gives the same within noise):**
  - 69.2% of starts at the right ayah within 5 s.
  - 10.3% of starts show a false lock first; none unresolved.
  - 1.76% of locked time on a wrong ayah; 9/100 jumps false.
  - Median lock delay after the distinguishing word: 0.76 s.
  - Below the PLAN bar (95% and 2%). The misses:
    - mujawwad elongation: one first word lasts 16 s;
    - al-Fatiha restarts in prayer, whose shared al-hamd opening needs 1:3;
    - repeated or basmala-prefixed Alif-Lam-Mim fitting 3:1–2 or 27:30–31;
    - the isti'adha matching 2:67.
- **Speed fixes:**
  - Global-search bins are now fixed-width around vote peaks. Chained diagonals had merged into one bin covering the whole Quran on long clips.
  - The DP inner loop uses table lookups.
  - Effect: a 424-phoneme clip went from 34 s to 0.32 s; shard-0 relabel takes 29 s.
  - Relabel labels are identical on all 970 clips. One clip lost a weak runner-up candidate.
- **Streaming acquisition:**
  - History is capped at 96 phonemes.
  - History is reset on `lost`, both while acquiring and while re-acquiring.
  - When the full history doesn't lock, the latest 48 phonemes are also tried.
  - These moved the QUD 5 s rate from 48.7% to 69.2%. Single-push relabel is unchanged.
- **Perf gate:** `pnpm perf` replays a fixed 260 s script covering acquisition, local tracking, a slip, noise, `lost`, re-acquire, a jump and silence, in 1 s pushes.
  - It fails if the real-time factor is over 1/speed-factor, or the slowest push is over 1000/speed-factor ms.
  - Default speed factor is 20. That's a guess for Hermes on the Android floor, until E7 device measurements.
  - Now: rtf 0.0014, slowest push 28 ms, pass. The worst QUD recording runs at rtf 0.0065.
- **Found:** the teacher's frame stride is 40 ms (measured: audio length over frames). `tools/make_fixtures.py` and `export_pass1.py` hardcode 20, so tlog event times are half the real time. Labels are unaffected.
- **Decision:** nurlingo was dropped from this ticket. It has no audio or phonemes, so the phoneme tracker can't consume it. "Both datasets" now means QUD only.
- **Frame stride fixed:** `tools/make_fixtures.py` now uses 40 ms, and the tlog fixtures were updated. The snapshots don't record times, so they're unchanged.
