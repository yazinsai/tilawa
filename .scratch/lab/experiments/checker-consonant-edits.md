# Count consonants replaced by vowels as hard edits

Status: implemented; practice result pending. User decides fold or discard.

Base: main 5716d49 (engine source identical to scored baseline b544715). Test set lab-v1.4.v3, practice only.

## Prediction before scoring

Three fewer checker-attributed and overall phone wrong calls: p10 26:106:5 and p10/p20 26:142:5. Preserve clean recitation and the studio copy. No acoustic threshold change.

## Data review and cause

All three labels have matching saved label-blind Soniox and Nemotron transcripts. The p20 substitution also has a saved human approval. No disagreement requiring a flag. Teacher agreement is only correlated evidence; the independent opinions were calibrated separately on human-reviewed practice clips.

The actual alignments exchange نوح and صالح. Because their lengths differ, some changed consonants align with vowels. subCharge only applies the substitution cost if both symbols are consonants, so those changes receive a soft pronunciation charge. Only one hard edit remains, below the two required for these words. Perfect transcripts and teacher evidence fail for the same reason.

Use the existing substitution cost for consonant/consonant and consonant/vowel pairs. Pure vowel changes, ghunna and recitation marks stay soft, and the existing confusable-consonant costs and posterior support requirements remain in force. No case-specific word rule.

## Validation

Two synthetic tests substitute نوح and صالح in both directions using the real reference phonemes and streaming checker. Both return no mistake on main and fail their expected wrong_word assertion. Red run: consonant-red-dc5b38b-dirty-20261030T175622Z.

The first, broader either-consonant prototype fixes these tests but fails four existing assertions: ghunna-only words and clean teacher fixtures. A focused real fixture replay shows رَببَهُ۾۾۾ heard as رقَهُم: the extra hard edit was the legitimate shafawi ghunna mark heard as م, alongside one actual consonant error. This is a pronunciation representation, not another word edit. The refinement explicitly limits the new charge to vowels, leaving those marks soft. This is the second implementation iteration; no practice score was run on the rejected prototype. The original hypothesis and failure are retained in the JSON card and ledger.

Prototype run: consonant-green-dc5b38b-dirty-20261030T175720Z. Clean fixture probe: consonant-clean-probe-dc5b38b-dirty-20261030T180054Z. Refined build and all 53 checker assertions pass on Modal in consonant-vowels-unit-dc5b38b-dirty-20261030T180231Z, both shards exit 0. The two target regressions pass and the four prototype failures are gone.

All tests, scoring and case replays run through lab run on Modal. No tracker, listener, scorer or test-set change.
