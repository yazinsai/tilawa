# Word-level CTC rescoring for wrong-word warnings

Status: killed. No engine behavior is retained.

- Experiment: `listener-ctc-rescore-wrong-words`
- Test set: lab-v1.5 practice (100 sessions); exam was not scored
- Prediction: listener −4, overall −4
- Result of the rejected prototype: listener −4, overall −2
- Practice scores used: one

## Mechanism

The proposed gate compared two forced CTC forward scores over the decoded word's frame span plus two frames: the written
phoneme sequence and the greedy decoded sequence. A warning would survive only when the decoded sequence beat the
written sequence by a fixed log-likelihood margin. This was intended to reject a short argmax spike while preserving a
real spoken substitution.

The prototype exported a generic CTC forward operation from the SDK and applied the same operation to the checker's
stored top-5 posterior rows. It removed three wrong-word false warnings, but added a phone miss and a studio miss. Its
overall phone change was −2, below the fold gate and within noise.

## Validation

The prototype's 14-nat margin was selected after scanning labeled practice outcomes. That is benchmark tuning, so the
prototype and its score cannot justify a production cutoff. The implementation has been reverted rather than assigning
the same value a post-hoc rationale.

The scan and score are still sufficient to evaluate the card's explicit kill criterion. Target false warnings removed
by the prototype had word-score gaps around 11–12 nats. Seven planted prophet-name swaps on perfect evidence had gaps
around 12.4–12.6 nats, and one planted swap was newly missed on studio evidence. The score therefore does not separate
the false-warning mechanism from real swaps: a margin that vetoes the observed false warnings also vetoes real spoken
substitutions.

The rejected prototype changed phone wrong calls from 89 to 87: three false warnings removed and one new miss. Studio
wrong calls increased from 58 to 60. Perfect-evidence wrong calls increased by seven, and no new false warnings appeared
on clean sessions.

## Limits

Top-5 rows discard the exact probabilities of unlisted written phonemes, so their forced score is partly an
approximation. Full-vocabulary logits could change calibration, but testing that after seeing the labeled practice
boundary would be another experiment with a margin fixed from independent data; it cannot rescue this card by tuning on
the same practice set.

The word span also misses dropped onsets when the greedy decode assigns no phoneme to those frames. A wider or
alignment-derived span is a distinct mechanism and would need its own hypothesis and predeclared decision rule.
