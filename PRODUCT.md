# Product

<!-- impeccable:product-schema 1 -->

Inferred from `PLAN.md`, `.scratch/lab/spec.md`, `GLOSSARY.md` and the lab redesign brief (29 October 2026); no interview was run. Facts marked *(inferred)* need confirming.

## Platform

web

## Users

- **The founder** (expert, product-minded) is the primary user of the lab pages. He checks whether the engine got better, decides to fold in or discard experiments, and listens to failing cases to tell bad data from engine faults. Short sessions at a laptop, often between other work *(inferred)*.
- **Agents** write, run and score experiments; they read the same numbers through the CLI and JSON, not the pages.

## Product Purpose

Tilawa is an offline streaming Quran recitation engine and SDK (Node, React Native, web) that follows free recitation and flags memorisation mistakes. The lab is its internal measurement surface: one overall score (wrong calls per 5 minutes on the exam set, lower is better), split into the listener, tracker and checker, plus experiments and a listening review of test data. Success: the founder knows in seconds whether the engine improved, which part to work on next, and which experiments need his call.

## Operating Context

- Served as one Modal web app (`modal/lab_review.py`) over the `tilawa-lab` volume; pages are static HTML/JS reading small JSON endpoints.
- Verdicts from the review page and the drill-down are written to the review store and become flags for `lab apply-flags`.
- Fold-in, discard and runs happen in the CLI (`python3 tools/lab.py`); the page shows commands to copy.

## Capabilities and Constraints

- Counts, not percentages, in the headline; a change of 2 or fewer wrong calls is noise.
- Exam cases are never shown; only practice cases can be opened.
- Terms follow `GLOSSARY.md` (Measurement); the page explains them in plain words.
- No auth; vanilla HTML/CSS/JS, no framework *(brief)*.

## Product Principles

1. One number first: the overall score is always visible and always explained.
2. Show where to act: the split must add up and make the biggest lever obvious.
3. Decisions over data: what needs the founder's decision stands out; raw numbers sit behind disclosure.
4. Listening is focused work: one case at a time, audio and Arabic text first.

## Accessibility & Inclusion

WCAG AA contrast; full keyboard use on the review flows; Arabic text set right-to-left in a proper Naskh face.
