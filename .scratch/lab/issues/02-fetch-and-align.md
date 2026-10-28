# 02 Fetch mp3quran surahs and align words

Status: ready-for-agent
Type: task
Blocked by: 01

Spec: `../spec.md` decisions 4 and 15.

A Modal job downloads the full-surah MP3s and ayah timings for the held-out reciters onto a lab volume, converts them to 16 kHz mono, and uses the teacher's CTC to force-align words inside each timed ayah.

## Acceptance

- Each surah file on the volume has ayah and word timings, plus the source URL, hash and API revision.
- Each ayah is checked against the alignment; any mismatch over 300 ms is listed and excluded.
- Nothing is downloaded or decoded on the Mac.
