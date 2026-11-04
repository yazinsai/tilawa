# Erasure-tolerant written hypothesis on the non-lexical fallback

Status: implemented and validated. Left for supervisor scoring and the fold gate.

## Preregistered prediction

Card `listener-erasable-written-fallback`, listener, explore, practice `lab-v1.5`.
Veto both target phone wrong-word false warnings, with about one caught planted swap lost:
listener −2, overall −1. The card is unchanged.

## Mechanism

The string-path lexical gate has a twice-`SUPPORT` fallback for a heard span that does not spell an attested
non-neighbour word and has no surplus consonants. Its written hypothesis requires every written phone to emit.
A consonant erased by the channel can therefore increase the heard-versus-written margin without supplying
evidence of a different consonant.

On that fallback only, reuse the checker's existing alignment of the heard visit onto its written word.
A written consonant deleted by the alignment, or aligned to a non-consonant, gets an alternative emission arc:
its own phone, blank, or an inventory short/long vowel (`َُِاۦۥ`). A written consonant aligned to a different heard
consonant remains required. Re-score the same full written window, including unchanged neighbours, through the
existing CTC forward sum. The greedy heard hypothesis and the twice-`SUPPORT` threshold stay unchanged.

These arcs make consonant emissions optional without skipping the label position: it still occupies a frame,
and the original CTC transitions, timing sum, and repeat arcs stay intact. Alternative token IDs are deduplicated.
Graph wins, lexical spellings, and the more-consonants escape do not get erasure arcs. The decoded string and
tracker are unchanged; the rule can only reduce the written hypothesis's negative log-likelihood and veto support.

## Validation

Modal run 5 compared the baseline checker from this branch's HEAD with the modified checker on the exact
card listener's stored phone evidence, `exp-listener-heard-word-must-be-lexical-41570706`. Both checkers received
the same chunks and position events, first with oracle positions and then with the ordinary tracker. These are
target replays, not a practice score.

| Target | Heard span | Erased written consonant | Original margin | Relaxed margin | Result |
| --- | --- | --- | ---: | ---: | --- |
| `lab-v1.5-p11/phone/warning:wrong_word/26:106:1/104.32` | `يَاا` | ذ, aligned to ا | 23.9774 | 7.6123 | confirmed → uncertain |
| `lab-v1.5-p36/phone/warning:wrong_word/55:48:1/282.64` | `َوَاافَا` | ذ, deleted | 20.4997 | 15.2505 | confirmed → uncertain |

Both margins are below the unchanged 20-nat bar: **2/2 targets vetoed**. The results are identical under oracle
positions and the ordinary tracker. Across each entire session, the only changed event is the target's final
state; all other event fields and all other checker events are byte-identical. The planted صالح-for-نوح swap
in the first session stays confirmed.

The actual alignment of إذ preserves ء→ي as substitution evidence; only ذ gets erasure arcs. Erasing ء is not
needed to veto the warning. The default `spliced-clean-int8` stored evidence differs from the card listener at
ذواتا: it still has a low-confidence د counterpart. Run 1's default-listener trace is therefore not the target
measurement; run 5 uses the listener named by the card.

Unit tests cover summed blank/vowel/phone arcs, duplicate emission IDs, unchanged ordinary and repeat sums,
contradictory consonants, and the requirement that remaining written labels still fit. Checker tests cover a
non-lexical visit with missing consonants and a non-lexical consonant swap, alongside existing lexical,
more-consonants, and parallel-phrase coverage.

Run 5's focused tests passed: four CTC tests and two checker tests. In the erasure fixture the original margin
is 26.5337 nats and the relaxed margin is −1.6079; the decisive non-lexical consonant swap keeps exactly the same
27.6516-nat margin, with no erased positions.

Modal run 6 validated the final tree with `bash tools/autolab_tests.sh` across six shards. Core and SDK builds
passed; 704 core tests and 16 SDK tests passed, with one SDK test skipped. All six shards exited 0. The checker
test shard hit Vitest's known `onTaskUpdate` RPC timeout after all 113 tests passed; the existing harness accepted
that specific infrastructure error and continued to the SDK tests, which passed.

No `practice-score` calls have been made. The card lists two targets, so its specific kill criterion (fewer than
two vetoed) was used for this target validation rather than the generic three-target instruction. The overall,
component attribution, studio, clean-session, reader-exclusion and incident-loss results remain for the supervisor.

## Limits

This does not verify that a missing consonant was actually recited. A planted swap heard with deleted consonants
can also benefit from the erasure arcs and lose its confirmation. Practice scoring and the supervisor's fold gate
decide that tradeoff. Target trace results alone are not an overall or attribution score.
