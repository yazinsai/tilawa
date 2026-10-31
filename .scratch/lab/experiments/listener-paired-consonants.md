# Paired clean/phone consonant preservation

Status: running; not folded.

Fixed on main cd6fb5d: seed 0, same 50h training pool, 200 updates, spliced-clean initialization, full size, learning rate .0015, batch budget 800s, spliced transcript CTC and KD2. Both arms add a clean view with transcript CTC to prior 75% phone-channel training. Candidate additionally uses weight .5 clean-stop-gradient-to-phone full-posterior KL, balanced by clean-argmax blank versus nonblank frames. Blank mass is retained. Control uses zero KL. No parameter sweep or canonical decoding.

Final scheduled checkpoints only: paired-consonants-v1 and paired-control-v1. Final int8 comparison uses lab-v1.5 practice phone/studio and current main checker. Disclose recovered and newly missed mistakes, removed and new warnings. A gain of two or fewer is noise. Require zero new false alarms and zero newly missed mistakes. No exam, fold, or label changes.

Real-mistake controls are held-out evaluation only: confirmed help omissions in muo5gkq2_857a61f4 (مع), muodfjiw_c16401bc (أخوهم only; unresolved نوح/هود excluded), and substitution in muodhmzj_0639dae3 (الجنة for الجحيم), plus reviewed realuser omissions. No benchmark audio is trained on. Clean-match teacher labels are provisional, not verified mistakes. Same-waveform pairs never replace recited errors with canonical audio.

Python compilation passes. Modal runs launched; results pending.
