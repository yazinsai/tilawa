# 11: Non-Quran phrases and basmala starts

**What to build:** The tracker recognises the fixed non-Quran phrases that surround recitation and doesn't try to locate them. A basmala tells it that a surah start is likely next.

- **Known phrases:** the isti'adha, takbir ("Allahu akbar"), "sami'a Allahu liman hamidah", "rabbana wa laka l-hamd", tasbih in ruku' and sujud, the taslim, and amin. Each is phonemized with the reference profile and kept in the index as a non-Quran phrase.
- **Skipping:** when recent evidence matches a known phrase better than any Quran location, it is consumed as non-Quran:
  - it never locks, jumps, or triggers `lost` on its own;
  - it's cleared from the acquisition and global-search windows, so it can't push the search toward a passage with text before the phrase (10:10 for al-Fatiha, 2:67 for the isti'adha);
  - a phrase that is also Quran text ("a'udhu billahi" in 2:67, "al-hamdu lillahi") counts as non-Quran only when it's recited whole and alone, between pauses or before a recitation.
- **Basmala as a surah prior:** a basmala that isn't continuing 1:1 or 27:30 gives a bounded bonus to the first ayahs of the surahs. It works like the existing `prior`: it can break near-ties but never overrides fit or separation.
  - After a basmala, "Alif Lam Mim" should stay `ambiguous` between its six surahs, never fit to 27:30–31 or 3:1–2.
  - While locked, a basmala at the end of a surah should hand off to the next surah's first ayah, without flickering back to the previous one.
- **Events:** no new event types. A skipped phrase may show as `acquiring` or hold the current position; trace records which phrase matched.

**Blocked by:** 10.

**Status:** ready-for-human

- [x] The isti'adha, then a basmala, then 2:1, never locks on 2:67, 27:30–31, or 3:1–2, and locks on al-Baqara once 2:2 distinguishes it.
- [x] A takbir before al-Fatiha in prayer doesn't delay its lock past the point where 1:3 distinguishes it.
- [x] "Alif Lam Mim" recited twice stays `ambiguous` and doesn't lock on 3:1–2.
- [x] Amin after 1:7 doesn't move the bookmark.
- [x] Surah-to-surah transitions with a basmala, as in Juz Amma, produce no false jumps and no flicker.
- [ ] Re-running the QUD benchmark improves the 5-second rate and false-jump rate over `docs/benchmarks/qud-v1-*` with no regression in the shard-0 relabel labels; the new report is committed.

## Comments

- **Phrase asset:** `tools/build_phrases.py` writes `assets/phrases-v1.json`. It phonemizes the isti'adha, takbir, tasmi', tahmid, ruku' and sujud tasbih, taslim and amin with the reference profile. The phonemizer misreads vowel-then-shadda, so the builder normalises to the reference's shadda-then-vowel order. The outputs match QUD decodes; the takbir is exact.
- **Opt-in:** `createTracker(index, { phrases })` turns it on. Relabel doesn't pass it, and phrase effects only act on earlier pushes anyway, so the shard-0 relabel labels are unchanged on all 970 clips.
- **Detection:** a phrase counts when it aligns to the latest phonemes at 0.3 weighted edits per phoneme or less, and its last two phonemes were heard. Without that second condition, a basmala missing its final mim was cut early and the mim left behind as junk.
- **While locked:** a phrase is skipped only if local search fit it at 0.25 edits per phoneme or worse; a better fit means it's Quran text. On a skip:
  - local search rewinds to before the phrase;
  - the global window restarts;
  - the next distant lock commits without waiting out `jumpPhonemes`.
- **While acquiring or re-acquiring:** the phrase and everything before it are dropped. A skipped basmala gives surah openings the `prior`-sized bonus.
- **Lock guard:** no lock while the latest phonemes are a phrase in progress (at least 12 phonemes and 40% of it), or a restart of the window's start. This stops "a'udhu billahi" locking on 2:67 mid-phrase. Shorter prefixes like "Allahu" are everyday Quran text and cost starts in the benchmark.
- **Repeat collapse:** in a stream, a block of 8–40 phonemes repeated at the window's end is collapsed to one copy.
- **Surah entry (all modes, not just with phrases):**
  - Local search charges 3 edits to enter the next surah, except into at-Tawba or within 40 phonemes of a skipped basmala.
  - Without this, a takbir after a surah's last ayah slid into the next surah's opening: 2:286 → 3:1–2 "Allahu…", 36:83 → 37:1.
  - A real continuation without a basmala is still caught as a jump by global search, since another surah is never "near".
- **QUD with phrases** (`docs/benchmarks/qud-v1-*-phrases`), against the committed baseline:

  | metric | baseline | with phrases |
  |---|---|---|
  | correct within 5 s | 27/39 | 26/39 |
  | false-lock-first starts | 4 | 2 |
  | false jumps | 9/100 | 8/91 |
  | wrong-locked time | 1.76% | 1.85% |

  - Witr: wrong-locked time 15.8% → 5.9%.
  - Qiyamah: its 75:1 start now within 5 s.
  - Afasy: the isti'adha lock on 2:67 is fixed.
- **Why the 5 s box isn't checked:** two baseline early locks on doubled openings were lucky fuzzy fits; mandour's doubled "Ta Ha" is now collapsed and waits for 20:2. In prayer, the bookmark now holds through the takbir instead of going `lost`. Al-Fatiha then waits on its shared al-hamd opening, and those seconds score as stale-wrong. Moving the bar further needs either a stale-aware metric or acquisition that weighs "just after a takbir" toward al-Fatiha. That's a judgment call, so this is ready-for-human.
- **Perf:** rtf 0.0016 on the gate; the worst QUD recording runs at 0.0083.
