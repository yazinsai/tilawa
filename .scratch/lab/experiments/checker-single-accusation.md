# One spoken edit, one complete accusation

Status: open; awaiting the full practice result and the user’s decision.

## Preregistered prediction

On lab-v1.5 practice, relative to main b5bcc95 (97 wrong calls: 72 misses and 25 false warnings), predict unchanged misses and 3 fewer false warnings, or 3 fewer overall wrong calls. Predicted checker attribution change is 0: the targeted false warnings are currently listener-attributed. No confidence threshold changes. The original card is `lab-experiment.json`.

## Mechanism

An insertion detected inside a word is re-split by `resplit()`, but the original containing Visit and alignment still include its speech. The same observed sounds can therefore make the containing word wrong and also produce an added-word accusation. This experiment reconstructs that bounded portion of the alignment as a complete word fit plus an edge insertion before judging candidates or confirming them.

A repair requires one internal insertion for the visit, a chronological consecutive reference neighbor without an intervening jump, an intact remaining word under the existing tolerance, no non-Quran phrase evidence, and exact ordered coverage of the replaced observed indices. Other visits and insertion blocks remain unchanged. The same reconstructed path feeds judgment and posterior confirmation. One pass per repair family bounds recursion; existing joint-addition repair remains enabled when it has not already run.

A genuinely wrong remaining word keeps the existing separate-error path. Initial/final run edges lacking a neighbor are left unchanged because moving internal speech to an unowned edge could lose an addition. No reference-word, session, reader, or passage special cases appear in runtime code.

## Validation before full practice scoring

- Baseline `single-accusation-red-3949fb9-20261031T113624Z`: the synthetic insertion reproduces two confirmed warnings for one edit. The second initial fixture had an invalid ayah endpoint; it was corrected before implementation validation.
- Initial repair `single-accusation-first-3949fb9-dirty-20261031T114043Z`: focused regressions plus existing addition/noise tests passed.
- Final `single-accusation-final-3949fb9-dirty-20261031T114159Z`: TypeScript build; 5 new tests, 8 existing addition tests, noise tests and checker groups. The new controls cover another surah and inserted phrase, a genuine wrong word plus a separate addition, clean speech, and a repeated boundary word.
- Supplemental `single-accusation-remaining-3949fb9-dirty-20261031T114210Z`: existing corrupted-evidence and final-lock tests passed.
- Independent static review cleared the final bounded repair after requiring the chronological neighboring Visit guard.

## Limits

This fixes duplicated ownership of an internal insertion. A replacement whose non-overlapping suffix is treated as a separate added word is not automatically folded into the replacement; that needs stronger evidence about a complete spoken-word boundary. The prediction is not revised if such a case remains. No full practice result has been read or used to tune this implementation.

The root agent will record the full practice result, old/new split, clean warnings, changed cases, and concentration checks here. No fold or discard is authorized.
