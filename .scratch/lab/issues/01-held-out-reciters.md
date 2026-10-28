# 01 Held-out reciters and the practice/exam split

Status: ready-for-agent
Type: task
Blocked by: none

Spec: `../spec.md` decisions 4 and 8.

List the mp3quran Hafs reciters that have ayah timings (`/api/v3/ayat_timing/reads`, 98 at last count). Remove every reciter who appears in any source the teacher or the student was trained on: the everyayah family, QUD, zaibihassan, muaalem-annotated, and anything named in the student manifests. tlog has no reciter names; record that as an unverifiable risk rather than a pass. Split the remaining reciters into practice and exam sets, balanced by speed and recording quality.

## Acceptance

- `docs/benchmarks/lab-reciters-v1.json`: held-out reciters with mp3quran ids, the reason each excluded reciter was dropped, and the practice/exam assignment.
- At least 10 held-out reciters in total, or a note explaining the shortfall.
