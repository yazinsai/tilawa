# Blank versus consonant acoustic experiment

Status: running; not folded.

Baseline: main cd6fb5d, lab-v1.5 practice, frozen checker/tracker. Baseline scorer run exp-checker-extra-segmentation-rebased-4001bd6-20261031T152054Z has 64 misses and 25 warnings. Main listener is spliced-clean-int8. No exam decoding, fitting, or label edits.

## Diagnostic before intervention

Nineteen listener-attributed false-warning incidents from the earlier fixed-listener baseline remain the diagnostic selection. The word windows and nearby phones are stored with complete paired top-5 frame traces in listener-blank-diagnostic.json. Align against the as-said word phonemes, not the intended Quran text; studio is a comparison, not independent audio truth. With the checker's exact consonant inventory, phone windows contain 21 deletions, 25 substitutions and 17 insertions, versus studio 4 deletions, 11 substitutions and 15 insertions. These are approximate local alignment counts, including boundary neighbors. They do not establish acoustic truth at each phone. The dominant extra pattern is lost/changed consonants, not extra consonants. Missing clusters stand out in p32, p45, p65. Most emitted true consonants as well as false ones occupy a single 40ms CTC peak; deleting one-frame peaks would destroy valid controls.

## Preregistered single candidate

When (and only when) the greedy winner is blank, choose the highest-probability consonant among the unchanged top-five posteriors if its original probability is at least 0.30 and at least 0.75 times blank. Otherwise preserve the greedy winner. Re-collapse CTC normally; no lexicon, Quran context, duration pruning, beam or case-specific rule. One candidate, no sweep. This tests whether a small near-tie rescue can repair missing evidence without inventing evidence. The intervention leaves all original nonblank peaks intact, but may join repeats across a rescued blank; gemination and real errors are mandatory measured controls.

## Acceptance

Report phone/studio incident-level recovered and newly missed mistakes, removed and newly invented warnings. Require zero new misses/warnings, and improvements beyond two-count lab noise. Measure short consonants (one-peak as-said controls), gemination, pauses, repeat/restart and planted wrong/added/omitted mistakes separately. Original acoustic files and labels stay unchanged. A failed narrow test is evidence for training, not permission to tune another decoder threshold.
