# 05: Scripted-error generator, recorder, consent

**What to build:** Ground truth with labels known by construction. Generated scripts with planted errors, a web recorder, and consent, ready for 5–15 volunteers.

**Status:** ready-for-agent

- [ ] Script generator in core, using the reference index. It plants about one error per 2–3 ayahs from the v1 memorization and pronunciation classes, from Juz Amma, Juz Tabarak and a few long surahs. The mix is weighted so each class reaches at least 50 positives across volunteers. Each script records its planted labels.
- [ ] One clean script per volunteer, with no planted errors.
- [ ] Web recorder: shows the script with planted errors highlighted, records in the phone browser, uploads to a private bucket (Modal volume or R2) with speaker ID, device and script ID.
- [ ] Consent form covering evaluation and commercial training use, retention, and deletion on request.
- [ ] Speaker split: designate the held-out eval speakers before any training uses the audio.
- [ ] Volunteer recordings never enter git; `.gitignore` covers any local download path, as it already does for `.scratch/clips/`.
- [ ] Recruitment (human): mixed gender, mostly non-Arab, several phones.
