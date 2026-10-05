# 05: Reranking with top-k posteriors

**What to build:** When evidence carries top-k posteriors, the tracker reranks its shortlist using CTC posterior scoring, following the input contract defined in 01. Without posteriors, results are identical to the phoneme-only path.

- **Scoring:** a CTC forward score of each candidate span's reference phonemes against the frames. It uses the explicit blank probability and the contract's rule for the mass top-k omits.
- **Placement:** reranking only reorders the shortlist and feeds fit and separation. It never adds candidates that retrieval didn't produce.
- **Comparison:** a flag on the fixture harness runs the same fixtures with posteriors on and off and reports the differences.

**Blocked by:** 03.

**Status:** ready-for-agent

- [ ] Fixtures without posteriors produce exactly the same events as before.
- [ ] Where decoded phonemes are mangled but the posteriors favour the true span, the true span is ranked first.
- [ ] Mismatched frame stride or token inventory is rejected.
- [ ] A comparison report is produced across the full fixture set.

## Comments

- Context from 04 (2026-10-05): verified clip 875 (low-volume audio) locks at confidence 0.11, just above `minLockConfidence` 0.08. Three of its four edits are elongations decoded short (ghunna `ںںں`→`ں` twice, one madd `ا`), not low-confidence phonemes, so posterior scoring should be checked on whether it recovers held-sound length. Ghunna-length cost was done in 04 (875 now 0.25); confidence-from-cost-only was tried and rejected, see 04.
