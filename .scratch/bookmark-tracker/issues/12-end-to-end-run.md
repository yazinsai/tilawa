# 12: End-to-end run

**What to build:** A repeatable test of the whole path on audio the tracker has never seen: real recitation audio, the Muaalem listener, streaming evidence, the tracker, then events checked against what was recited.

**Blocked by:** 11.

**Status:** done

- [x] Sessions are assembled from everyayah per-ayah recordings by four reciters, with a truth timeline of what plays when.
- [x] The audio is decoded by the teacher on Modal, the way the listener would.
- [x] Evidence streams into the tracker in 1 s chunks. Each session is checked:
  - every passage start is located within 5 s, or within 3 s of when its audio allows a lock;
  - no wrong lock is shown first, and there are no false jumps;
  - the final position is right;
  - in memorization mode, the expected bookmark stays on recited ayahs once a jump commits.
- [x] All six sessions pass.

## How to run

```sh
uv run --python cpython-3.12-macos-aarch64-none --with numpy python tools/build_e2e.py   # .scratch/e2e/*.wav + truth
modal run modal/decode_audio.py                                                          # *.evidence.json
pnpm --filter @tilawa/core e2e [--verbose]                                               # exits 1 on any failure
```

Sessions are defined in `tools/e2e_sessions.json`.

## Comments

- **Sessions:** 18 minutes of audio across six sessions:
  - isti'adha, basmala, then Ya-Sin (Alafasy);
  - Ayat al-Kursi, a pause, then a jump to al-Mulk, in free and memorization modes (Husary);
  - al-Fatiha, then al-Ikhlas, al-Falaq and an-Nas, each after a basmala (Abdul Basit);
  - al-Kahf, 8 s of noise, then ar-Rahman (Minshawi);
  - a cold start at 59:22 (Ghamadi).
- **Tracker fixes the run found:**
  - **Basmala after al-Fatiha:** the basmala opened the next surah in local search, so 112:1 was fit into 2:1 for 4 s. A basmala skipped while locked now hands over to global re-acquisition. Surah openings are favoured, with the next surah slightly more, and the last position stays on screen meanwhile. The next surah's opening locks as soon as it's the top candidate, except after al-Fatiha. In memorization mode, anything other than the next surah's opening freezes the expected bookmark, like a jump.
  - **Memorization freeze point:** it was captured after local skips had already drifted (2:259:39). Recent cost lags a deviation by about 8 phonemes, so it's now taken from 8 phonemes earlier: 2:257:24, the last recited word.
  - **Jump onsets:** a new local run (skip or restart) now shows only once its last 3 phonemes cost at most 0.34 edits each. This cut the flashes into wrong ayahs at jump onsets.
- **Distinguishing point:** truth here is per ayah, so the point where the audio allows a lock is interpolated by phoneme position. That's off by up to about 2 s, hence the 3 s margin.
- **Regression checks after these fixes:**
  - unit tests: 378 pass;
  - relabel: shard-0 labels unchanged;
  - perf gate: passes;
  - QUD with phrases: 30/39 starts within 5 s, 2 false-lock starts, 8/91 false jumps, 1.83% wrong-locked time (1.72% before these fixes).
- **Known:** the QUD Juz Amma recording still shows the next ayah's first word 1–2 s early at rhymed ayah endings. That's local search pre-empting a rhyme, and it predates this ticket.
