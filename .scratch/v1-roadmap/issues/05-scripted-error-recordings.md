# 05: Gap recordings: pairs, shadda, long sessions, self-correction

**What to build:** Scripted-error recordings for what the existing help-page clips (ticket 12) don't cover. They supply ≥ 50 positives for the memorization classes and short vowels, plus 464 clean clips. This ticket fills the rest.

**Blocked by:** 12 (so the audit numbers are final before scripts are weighted).

**Status:** needs-triage

## What the help clips lack

- Letter pairs: only ق/ك (11 clips from 2 sites: خلق and الفلق). There are also 15 "light ص" clips from 2 sites (الصمد and صدور), which are weakened emphasis, not a clean ص→س swap. The other 7 pairs have none.
- Missing shadda: 5 clips, all in الخنّاس.
- Long sessions: clips average about 11 s, mostly a single ayah that contains the error. Nothing tests staying quiet across minutes of clean recitation between errors.
- Self-correction: no scenario says a mistake and then fixes it (`hesitate-*` are repeats).
- Speakers: about 18 clips from women, and native language was never asked.
- Short vowels: concentrated on about 4 v2 sites, mostly with fatha as the original vowel. Damma and kasra as the original vowel have ≤ 3 clips each.

## Tasks

- [ ] Pilot ahead of the full script, needed by ticket 11: 5–10 real clips per direction for ت→ط, ذ→ز, س→ث, ز→ذ and ث→س. Each clip is the word recited in context with the wrong letter said on purpose (e.g. ط in place of ت). Splicing couldn't settle these directions: the splices are jarring, and the teacher fills in the canonical letter. Score the pilot with `tools/e1_report.py` before weighting the full script.
- [ ] Count pair positives per direction, not per pair. E1b (ticket 11) shows the teacher hears one direction far better than the other (ط→ت 70% vs ت→ط 23% on splices).
- [ ] Script generator in core, using the reference index. Long multi-ayah scripts from Juz Amma, Juz Tabarak and a few long surahs, with about one planted error per 2–3 ayahs. Weight the mix toward the gaps above until each pronunciation class has ≥ 50 positives from ≥ 15 distinct sites. Each script records its planted labels.
- [ ] Self-correction items: say the error, then re-say the word or ayah correctly. Label both the error and the fix.
- [ ] One long clean script per volunteer, for false warnings per 5 minutes on real session lengths.
- [ ] Recorder: extend the existing help page (`offline-tarteel/web/frontend/server/help`) with a v5 prompt set rather than building a new one. Keep the same storage and export, and add the script ID and planted-label JSON per recording.
- [ ] Add a native-language field (Arabic / not) to the speaker form.
- [ ] Consent: keep CC BY 4.0, and add retention and deletion on request. Deletion needs a speaker-held deletion code, because emails are not linked to recordings.
- [ ] Recruitment (human): women and non-Arab reciters first, several phones.
- [ ] Recordings stay out of git.
