# 21: Real-user false warnings added by three checker folds

Status: ready-for-agent

Found 5 Nov, once the realuser slice joined `lab-v1.5u` (ticket 19). The main line gives 7 false warnings in 63.3 recited
min of real learners' practice audio, against 1 at autolab start (`abb918b`). Each main line was scored on `lab-v1.5u`
practice with its own listener evidence, without touching the ledger:

| main line | fold | practice | realuser FW | added |
|---|---|---|---|---|
| abb918b | autolab start | 89 | 1 | sob-049 92:13:1 (known tajweed slip, ticket 19) |
| d5303ef8 | listener-context-free-gap-decode | 84 | 1 | |
| 37712511 | checker-dropped-suffix-wordform | 80 | 3 | sob-053 88:2:2, tlo-127 2:29:16 |
| d4ec2103 | checker-parallel-verse-graph-decode | 73 | 4 | sob-027 84:25:1 |
| 3bbe6794 | listener-heard-word-must-be-lexical | 51 | 4 | |
| 75fde7d1 | checker-ledger-with-without-ayah-fit | 46 | 7 | ret-017 110:2:1, sob-034 79:29:1, sob-034 79:44:1 (skips) |
| 2b50d856 | today | 30 | 7 | |

## What each one is

- **sob-053, tlo-127 (dropped-suffix-wordform).** The expected form opens with the consonant doubled from the previous
  word's tanween (يييَومَءِذِن after وجوهٌ, ووَهُوَ after سماواتٍ). The learner paused between the words and started
  the second plainly, as is correct after a stop. `index.isolatedWordForms` already holds the plain form, but only the
  pausal-ending paths use it (`checker.ts` pausalManner, pausalVisit). Fix: after a heard pause, accept the isolated
  onset as the written word.
- **sob-027 (parallel-verse-graph-decode).** إلّا heard as إنّ at 84:25:1. Not yet looked at.
- **ret-017, sob-034 ×2 (ledger-with-without-ayah-fit).** Skips of whole ayahs that were recited. sob-034's two are also
  raised by the teacher and the oracle-evidence checker, so the ledger fit is wrong on real audio, not the listener.

## Next

Each is a checker experiment (`lab new`), scored on `lab-v1.5u`, whose gate now counts real-user false warnings. Reverting
the three folds instead would give back about 16 practice wrong calls.

Scoring script: `score_old_main.py` (session scratchpad), runs `lab.score` on a detached worktree.
