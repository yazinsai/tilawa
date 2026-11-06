# Wrong-word rival locality

Status: broad veto scored worse; narrowed to ayah-initial slots, validation pending.

## Preregistered prediction

Experiment `listener-swap-rival-surah-locality`, listener, explore. Practice set `lab-v1.5u`, listener
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. Predicted 2 of 4 target phone false warnings removed,
about one caught swap lost: listener −2, overall −1. The card is unchanged.

The task requires stopping if the built veto removes fewer than three registered targets, even though the card
predicts only two. The card's separate premise kill is fewer than two targets with rivals absent from both links.

## Mechanism built

The initial broad build applied at final confirmation to every non-graph wrong word. It retained its existing acoustic and lexical tests and additionally
requires its attested rival to have a memorisation link. A rival is linked if its mushaf word form occurs in the
current surah or in the existing folded parallel-phrase entry for the accused slot. Letter identity uses the
existing diacritic/alef folding; supported phone forms map back to all their attested word identities.

For a lexical candidate the named pronunciation stays the rival. For an ordinary string-path spelling the
existing whole-word lexical test supplies the rival. If the span is non-lexical, all whole attested forms with at
least two consonants compete under `spanLogP` and the unchanged `LISTENER_CHANNEL`, after held-sound collapse.
The written word, nearby words, prefixes and contiguous neighbourhood runs are excluded using the existing
lexical neighbourhood test. Locality is checked after ranking, so a local word cannot displace a better distant
rival. Missing word text, an empty channel for a non-lexical span, or an unidentified rival abstains.

Parallel-graph wins are exempt. No constants, model weights, decoded tokens, tracker events, or word boundaries
change. The rule changes only a pending confirmation to uncertain.

## Validation of the broad build

Targeted engine and true-position replays compare the built checker against the branch's original checker on
the same stored phone evidence. Diagnostics record the original confirming path, heard span, chosen attested
rival, and both locality links. These are target diagnostics, not a practice score.

Modal run 4 passed all 11 focused unit and checker tests (1.09 s). The first test run passed 10 and failed one
fixture that accidentally placed the rival inside the existing ±3 exclusion; the corrected fixture places it
outside that neighbourhood and asserts its identity as well as its locality. No engine rule changed for that fix.

Unit tests cover same-surah rivals, cross-surah parallel completions, distant rivals, slot-specific parallel
links, non-lexical channel ranking, preservation of a named lexical rival, missing-data abstention, clean text,
and checker confirmation of local and parallel substitutions.

Modal run 6 replayed all four complete sessions using the exact card listener's stored phone evidence.
Each target warning was present in the original checker and absent with the veto, in both the free-tracker
and true-position arms. All rivals were absent from both locality links:

| Slot | Heard span | Best attested rival | Target result |
| --- | --- | --- | --- |
| 26:106:1 | `يَاا` | `يُرَا` (يرى) | removed |
| 55:48:1 | `َوَاافَا` | `يَخَافَا` (يخافا) | removed |
| 26:174:8 | `مُءمِنِۦۦنقَلِۦلُمن` | `بِطَرِۦقَتِكُمُ` (بطريقتكم) | removed |
| 92:13:1 | `وَءِليَ` | `وَءِلَييَ` (وإليّ) | removed |

The four replays each lost exactly that one confirmed warning and gained no confirmed warning, in either arm.
This is a four-session diagnostic comparison, not a full practice score or a planted-hit census.
The original checker source was captured byte-for-byte from starting commit `6db04f77`;
production code and focused tests were committed as `b3869000`.
Raw records are `.autolab/modal-out-6/{0,1,2,3}/result.json`. Temporary replay helpers were removed.

Modal runs 3 and 5 did not execute the replay: the snapshot omits `.scratch` files and the
experiment card. Run 6 uses temporary core-test helpers and passes the card listener name explicitly. The helper could explicitly fall back to shipped evidence, but no shard needed that fallback. Practice score 1 of 3 was requested after three shards established the continuation minimum. Results pending.
Modal full-suite runs 7 and 8 both built core and SDK successfully, then failed the same 17 core tests:
9 checker event/confirmation/corruption tests, 3 manner tests, 2 acceptance tests, 2 lab-engine attribution
tests, and 1 lexical substitution test. The veto suppresses existing deliberately substituted words, including
clear distant-surah words and a same-surah Hood-for-Noah control whose string-path rival ranking does not retain
that local identity. SDK shards following failed core shards did not run; run 10 below completed the full SDK suite.
Run 8 also hit Vitest's worker RPC deadline on the checker shard, alongside substantive assertion failures.
These failures are not treated as a premise kill. The practice result is required before narrowing.

## Limits

This is a text-locality prior rather than additional acoustic evidence. Deliberate substitutions from unrelated
surahs can be vetoed; a phone blur that resembles a word in the current surah remains eligible. Homophonous
attested word forms retain every identity, so any local identity keeps the accusation. The existing parallel
index describes four-word phrases with an interior difference; it does not enumerate every hifz association.

Generic synthetic diagnostic, Modal run 9: the existing same-surah نوح→هود acceptance test is vetoed because
`surplusWord` first returns the vowel-soft spelling `هُۥدَن` (هودا) for the heard `هُۥۥدُن` (هود). The selected
representative is absent from surah 26, although the exact heard pronunciation maps to هود, which is present.
Thus a lexical spell match can have multiple word identities; choosing its first soft pronunciation is not
necessarily choosing the best attested rival. This is separate from the intended veto of الكوثر for يشفع,
which the same diagnostic also reproduces. Raw observations are `.autolab/modal-out-9/0/log.txt`.
The scored engine was not changed during this diagnostic.

Modal run 10 separately rebuilt core and SDK and ran the complete SDK suite: 25 passed, one existing skip.
The standalone run was necessary because the full-test wrapper stopped before SDK execution on the core
shards with failures. Core's broad-veto compatibility failures remain pending resolution.


## First practice score and narrowing

The supervisor scored `b3869000-dirty` once on the card's practice set:

| Phone practice | Main | Broad veto | Change |
| --- | ---: | ---: | ---: |
| Overall wrong calls | 24 | 44 | +20 |
| Missed mistakes | 13 | 36 | +23 |
| False warnings | 11 | 8 | −3 |
| Listener attribution | 23 | 32 | +9 |
| Tracker attribution | 1 | 1 | 0 |
| Checker attribution | 0 | 11 | +11 |

The supervisor verdict is worse. Raw summary: `.autolab/practice-result-1.json`.
The four diagnostic wins include a real-user session; the practice headline shows three fewer false warnings.
This summary does not provide the studio or clean-session incident lists, so no result is claimed for those gates.

Only after this score failed, the production scope was narrowed to ayah-initial words (`ReferenceWord.word === 1`).
Rival selection and locality membership are unchanged. Interior and ayah-final substitutions keep their previous
confirmation. The three initial-slot targets remain eligible; the glued-surplus target at 26:174:8 is deliberately
outside the narrower rule. The locality prior is now confined to the onset context behind three registered blurs,
rather than also suppressing arbitrary substitutions and surplus handling throughout a passage.
This is a structural scope change, not a margin change or a fitted numeric threshold.

The synthetic locality fixture now places its accused word at the start of a second ayah with an earlier anchor.
A new regression checks that a remote substitution inside the ayah still confirms. Existing tests are unchanged.
Modal run 11 passed all 12 onset/interior-focused tests. A further identity regression was then added:
when the existing lexical matcher returns an earlier soft-vowel spelling but the actual heard pronunciation
is itself attested, the locality lookup uses that exact pronunciation. It does not alter the shared lexical
matcher, token string, or channel ranking of a non-word. Named lexical candidates retain their named identity.
Modal run 9 supplied the control: exact spoken Hood was named as accusative `هودا` by the soft spelling test,
which has a different folded written identity and occurs outside the current surah. Prioritising an exact
pronunciation prevents that avoidable word-identity error. The regression uses synthetic case endings,
not that verse or any target reference.

Modal run 10 independently passed the full SDK suite: 25 tests passed and one existing help-e2e test skipped.
The final core/SDK suite and a second practice score are being requested for the narrowed build with the identity fix.
