# Lab: one score, three components, many experiments

Status: ready-for-agent

Agreed 28 October 2026 (grilling session). Terms follow `GLOSSARY.md` (Measurement section).

## Problem

The scoreboard has 13 release gates and about 200 diagnostics across five test sets. The shipped listener's latest row is mostly "not measured", and no number says whether the engine got better. There's no test of long free recitation with mistakes, no way to tell bad data from engine failures, and experiments are commits judged against whatever ran before. There is also a strong suspicion that both the tests and the training data are bad.

## Goal

- One **overall score**, always visible: wrong calls per 5 minutes of free recitation on the **exam set**. Lower is better; 1 miss = 1 false warning. The missed and false-warning counts are always shown under it.
- Three **component scores**, each with perfect upstream inputs:
  - listener: PER against what was actually said;
  - tracker: share of recitation time on the right place, counted from the disambiguation point, with false jumps;
  - checker: wrong calls per 5 min with oracle evidence and the true position.
- **Attribution** splits the overall wrong calls into listener, tracker and checker losses, which add up to the total.
- Many **experiments** in parallel. Each shows its component, the expected effect, the actual effect (component and overall) and a verdict. You fold in or discard.
- Every failing case can be opened (audio, decode, tracker, checker) and marked as either bad data or a component fault.
- **Data first.** No score is trusted until test set v1 passes review.

Non-goals for now: pronunciation (its own track once the checker has a class), the memorization-session use case in the headline, migrating old scoreboard rows.

## Decisions

1. The lab replaces the scoreboard front page. Release gates move to a secondary release tab. Old `scoreboard.json` history is archived read-only.
2. The headline use case is **free recitation** (no passage given), on minutes-long sessions so the tracker has time to lock.
3. Shared runs (Al-Rahman and similar) aren't excluded. Lock delay counts from the **disambiguation point**, and a confident wrong lock before it is a false jump. Most sessions start where the disambiguation point comes within about 5 words; a labelled **strict slice** starts inside long shared runs.
4. **Test set v1 = spliced sessions** from mp3quran full-surah Hafs recordings with ayah timings (98 reciters), using held-out reciters only. Word timings come from teacher alignment.
   - Size: about 40 sessions of 3–8 minutes from 10 or more reciters. About 20% start at an ayah 1, the rest at chosen ayahs.
   - Mistakes: about 1 planted mistake per 2–3 ayahs (wrong, omitted, added, skipped passage; mutashabihat swaps as their own slice), 150 or more in total.
   - Non-errors: about one planted repeat or restart per session.
   - About 25% of sessions are fully clean.
5. Mistakes are planted only at word boundaries with no assimilation across the words, with about a 10 ms crossfade.
6. The headline is scored on a **phone-degraded** copy of each session (room reverb plus licensed noise, fixed seed). Studio audio is a slice.
7. One test set feeds the overall score, all component scores and attribution. Gold PER and the help clips move to the release tab.
8. The test set is split by reciter into a **practice set** (fully visible) and an **exam set** (numbers only, scored at fold-in).
9. The **main line** listener is `spliced-clean-int8`. The teacher is shown as a ceiling only.
10. An **experiment** = a git worktree plus a card:
    - which component it's for;
    - the hypothesis;
    - the expected change in the component score and the overall score, in counts.

    Agents write, implement and launch experiments; you decide fold-in or discard.
11. **Fold-in** = rebase onto the main line and re-score automatically. The card shows expected, measured on its branch, and measured after rebase. A gain that disappears after rebase marks the experiment "superseded".
12. Changes are shown in counts. A change of 2 or fewer is noise.
13. **Flags** are pending until applied. Applying them creates the next test set version and re-scores the main line and all open experiments.
14. The drill-down surfaces likely bad data first:
    - the teacher, the student and the oracle-evidence checker all fail the same case;
    - the listener hears neither the original nor the planted word at a splice;
    - a case fails in every experiment.

    One click marks a case as bad data (label, audio, splice) or as a component fault (listener, tracker, checker).
15. **Modal by default.** Data building, decoding, scoring and the dashboard (a Modal web endpoint) all run on Modal. The Mac runs git and the CLI only.
16. **Test set acceptance.** Every automated check passes, including the seam gate (`lab_sessions.SEAM_LIMITS`: seam levels, clicks, seam depth, words found and words cut short, enforced in the build loop, which refuses and rebuilds failing splices). From lab-v1.2 the gate and the planner also hold three rules from your listens to lab-v1.1: a linked cut must be at least 6 dB below the ayah level in the source (8 dB after a vowel or long vowel), a wrong or added word lasts at least 0.26 s, and a cut before a word that starts weak (hamza, including hamzat al-wasl after a pause, ع, ح, ه, and خ, غ) must sit after at least 80 ms of measured silence (below the recording's room tone + 3 dB, never above 30 dB under speech). Then two-model triage (decision 17) writes the listening queue, and the set freezes once you have reviewed it with at most 1 bad per 40 items (`maxBad = max(1, items // 40)`). More bad than that means fixing the pipeline and rebuilding, not patching cases. lab-v1.1 was accepted on the earlier fixed check: 40 listens (20 splices, 20 random cases), at most 1 bad (`review/v1-check.json`, still built).
17. **Two-model triage** (ticket 17, `docs/benchmarks/lab-triage-v1.md`). Nemotron 3.5 Transcribe (LM pairwise) and FastConformer (CTC pairwise), both calibrated on the 40 listens (ticket 12), score every case: each planted edit against the original text at that spot, each repeat or restart with and without it, and each clean window of about 8 s against a one-word substitution or insertion. Each uses a 2-nat margin. If both say the label fits, the case is auto-accepted. If both say it doesn't, the case is queued for you to confirm; nothing is flagged or rebuilt automatically. Disagreement or an unsure model also queues it. On top, a fixed seeded spot check of the auto-accepted cases, 15 edits and repeats plus 15 clean windows, and a seeded seam spot check of 10% of the planted edits, at most 20. (lab-v1.1 used a seeded 10% of the auto-accepted cases: 144 of its 257 items, mostly clean windows both models had accepted.) A verdict from an earlier test set carries over only to an item whose clip bytes and text are identical.
18. The old help clips get the same automated checks plus a 20-clip listen. They return as a real-learner slice if they pass, and are dropped if they fail.
19. **Training data** gets its own tab after v1: about 30 sampled clips per bucket and source, an estimated bad-label rate with a range, and flags feeding a training data version. Retraining on it is a listener experiment.
20. Public error sets (Al-Ikhlas from Mendeley, RetaSy, QuranMB with its approved ground truth) go onto the Modal volume now. They aren't scored until there's a pronunciation track; RetaSy's wrong-verse labels later become a tracker slice on learner audio.
21. AGENTS.md: any change that can move a metric is an experiment with a recorded result, and fold-in updates the main line.

## Order

Data first (tickets 01–06, then your check in 07). The scorer, runner, ledger and dashboard (08–11) are built alongside, but their numbers aren't trusted until 07 passes. Then come review_asr calibration, the help clip audit, public sets, the training data tab and scoreboard retirement (12–16).
