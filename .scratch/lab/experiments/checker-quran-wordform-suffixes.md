# Attested word-form endings

Status: implementation revised after review; not practice-scored and not folded.

## Preregistration

`checker-quran-wordform-suffixes`, checker, explore.

Prediction: checker −5, overall −4. A suffix-only swap can be absorbed as waqf or held-vowel variation because it
does not substitute a consonant and may remain below the ordinary hard-edit threshold.

## Mechanism

The checker builds a vocabulary from every supported connected and isolated pronunciation in the reference. Each
collapsed form is indexed by stems obtained by removing a short tail. A heard word is an ending alternative when:

- it and a written form share a longer stem than either tail;
- exactly one tail contains consonants;
- a different attested form sharing that stem fits the heard word at least one hard edit better; and
- the written word's own pause or vowel-only forms do not explain the observation.

Only an added, decoded consonant-bearing suffix becomes a checker candidate. Its decoded frames provide both confidence
and a local CTC comparison between the written and heard forms. A missing suffix has no decoded consonant frame to
anchor the decision, so this implementation abstains instead of treating an acoustic deletion as proof of a word swap.
The following visited word establishes the right boundary; its exact identity is not a condition.

## Validation

Generic synthetic-reference tests cover:

- attested consonant-bearing suffixes in both lexical directions;
- checker detection when the suffix consonant is present in the decode;
- abstention for a missing suffix, absent or neutral posteriors, pause forms, held vowels, unattested single consonants,
  and consonant-free endings;
- unchanged clean passages.

The focused test passes. The full core suite passes in two Vitest halves, the SDK suite passes, and both packages build
(`adhoc-c5f854a-dirty-20261101T060405Z`). The unsplit core run also passed all 618 assertions but hit Vitest's worker-RPC
timeout after execution, so it is not counted as the green run.

Three practice runs had already been consumed by the rejected prototype. The broad attestation rule produced many new
warnings from listener consonant deletions. Narrowing it by particular vowel content and requiring an exact following
word made the result benchmark-shaped; both predicates and all benchmark-specific tests were removed. The revised diff
has not been practice-scored because the experiment budget is exhausted, so no fold-gate claim is made.

## Limits

- Missing suffixes remain undetected without an independently localizable consonant observation.
- The short-tail bound is structural but still excludes longer clitics.
- Reference attestation establishes that another word form exists; it does not perform morphological analysis.
