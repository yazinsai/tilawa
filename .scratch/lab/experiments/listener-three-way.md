# Three listener experiments — 31 October 2026

Status: complete; none recommended for adoption. No fold or discard performed.

All three experiments ran in separate agent-owned worktrees. Main reference is
`cd6fb5d` (engine source `4001bd6`), using the frozen lab-v1.5 practice set of
100 sessions, the same checker/tracker, and deployed int8 evidence. Baseline is
64 missed mistakes + 25 false warnings on phone, 45 + 13 on studio. No exam
scoring or benchmark training was used. The targeted development audit initially
decoded one gold clip; it was excluded from the reported development results
and did not select training examples, weights, parameters or checkpoints.

## Incident outcomes against main

| Experiment | Audio | Recovered mistakes | Newly missed | Removed warnings | New warnings | Misses after | Warnings after |
|---|---|---:|---:|---:|---:|---:|---:|
| Narrow blank rescue | Phone | 0 | 3 | 3 | 1 | 67 | 23 |
| Narrow blank rescue | Studio | 0 | 1 | 0 | 0 | 46 | 13 |
| Paired consistency | Phone | 15 | 21 | 12 | 17 | 70 | 30 |
| Paired consistency | Studio | 7 | 27 | 8 | 23 | 65 | 28 |
| Targeted weighting | Phone | 17 | 10 | 11 | 11 | 57 | 25 |
| Targeted weighting | Studio | 5 | 6 | 2 | 4 | 46 | 15 |

All incident deltas reconcile with totals. Comparisons match warnings
one-to-one by session and Quran word within six seconds, using the historical
listener comparison policy. Changed warning classes are retained separately.
`tools/lab_compare_incidents.py` selects practice records explicitly and checks
frozen labels, session metadata, and case-versus-session counts. It reproduced
the earlier full-phone experiment's counts and passed a baseline identity check.

## What the experiments establish

**Blank rescue fails.** It reduced phone phoneme errors from 9,039 to 8,475,
but recovered no missed mistakes. Genuine consonants commonly occupy one 40 ms
CTC peak, so peak suppression is not justified. Fourteen of 18 distinct missing
consonant/incident pairs never reached probability 0.30 in their word window.
Only one had a blank frame eligible for the frozen rescue rule, and that was a
repeated consonant already emitted elsewhere. This favors improving acoustic
evidence over another near-tie decoder adjustment. These are alignment/top-five
posterior diagnostics, not independent listening judgments.

**The paired training recipe fails.** Both the consistency candidate and its
matched two-view transcript-only control were much worse than main. Phone
wrong calls were 100 with consistency, 101 without it, versus main's 89;
phoneme errors were 28,055, 27,314 and 9,039 respectively. The one-call difference
is noise. This does not isolate consistency as the cause: the shared addition
of clean-view transcript loss also changes the training objective balance.

**Targeted weighting is the best aggregate direction, but does not earn
adoption.** It reduced phone phoneme errors to 7,674 and wrong calls from 89 to
82, while introducing 11 phone warnings and worsening studio results. Against
the earlier uniform full-phone model, it recovered three phone mistakes,
newly missed one, removed one warning and introduced none; studio recovered two
mistakes, lost none, removed two warnings and introduced four. That is evidence
worth retaining, not a safe deployment result. Mining selected 60.2% of clips;
94.8% received some upweighting, so this was broad relative weighting rather
than a small hard-example subset. Selection was based on model disagreement,
not verified false evidence.

## Controls and limits

The decoder retained original short-consonant peaks but worsened some geminated
words, repeats, restarts and pause-region emissions. Both training recipes kept
the detection outcomes of five reviewed real-mistake sites, including two
already-caught natural omissions. The paired model retained the actual
الجنة substitution instead of restoring الجحيم, but shortened repeated n
emissions, so gemination preservation is unproven. No verified natural
self-correction controls were available; phoneme-repeat proxies do not fill
that gap. Unreviewed warnings in the small real-user control recordings remain
unresolved, separate from the frozen practice false-warning labels.

## Detailed evidence

- [Blank diagnostic, controls and verdict](/Users/rock/ai/projects/tilawa-lab-exp-listener-blank-margin/.scratch/lab/experiments/listener-blank-margin.md)
- [Blank incident records](/Users/rock/ai/projects/tilawa-lab-exp-listener-blank-margin/.scratch/lab/experiments/listener-blank-comparison.json)
- [Paired training and matched control](/Users/rock/ai/projects/tilawa-lab-exp-listener-paired-consonants/.scratch/lab/experiments/listener-paired-consonants.md)
- [Paired incident records](/Users/rock/ai/projects/tilawa-lab-exp-listener-paired-consonants/.scratch/lab/experiments/paired-vs-main.json)
- [Targeted training and control](/Users/rock/ai/projects/tilawa-lab-exp-listener-targeted-consonants/.scratch/lab/experiments/listener-targeted-consonants.md)
- [Targeted incident records](/Users/rock/ai/projects/tilawa-lab-exp-listener-targeted-consonants/.scratch/lab/experiments/targeted-vs-main.json)
