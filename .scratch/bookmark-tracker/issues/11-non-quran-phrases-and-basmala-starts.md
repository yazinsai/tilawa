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

**Status:** ready-for-agent

- [ ] The isti'adha, then a basmala, then 2:1, never locks on 2:67, 27:30–31, or 3:1–2, and locks on al-Baqara once 2:2 distinguishes it.
- [ ] A takbir before al-Fatiha in prayer doesn't delay its lock past the point where 1:3 distinguishes it.
- [ ] "Alif Lam Mim" recited twice stays `ambiguous` and doesn't lock on 3:1–2.
- [ ] Amin after 1:7 doesn't move the bookmark.
- [ ] Surah-to-surah transitions with a basmala, as in Juz Amma, produce no false jumps and no flicker.
- [ ] Re-running the QUD benchmark improves the 5-second rate and false-jump rate over `docs/benchmarks/qud-v1-*` with no regression in the shard-0 relabel labels; the new report is committed.
