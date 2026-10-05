# 05: Reranking with top-k posteriors

**What to build:** When evidence carries top-k posteriors, the tracker reranks its shortlist using CTC posterior scoring, following the input contract defined in 01. Without posteriors, results are identical to the phoneme-only path.

- **Scoring:** a CTC forward score of each candidate span's reference phonemes against the frames. It uses the explicit blank probability and the contract's rule for the mass top-k omits.
- **Placement:** reranking only reorders the shortlist and feeds fit and separation. It never adds candidates that retrieval didn't produce.
- **Comparison:** a flag on the fixture harness runs the same fixtures with posteriors on and off and reports the differences.

**Blocked by:** 03.

**Status:** done

- [x] Fixtures without posteriors produce exactly the same events as before.
- [x] Where decoded phonemes are mangled but the posteriors favour the true span, the true span is ranked first.
- [x] Mismatched frame stride or token inventory is rejected.
- [x] A comparison report is produced across the full fixture set.

## Comments

- Context from 04 (2026-10-05): verified clip 875 (low-volume audio) locks at confidence 0.11, just above `minLockConfidence` 0.08. Three of its four edits are elongations decoded short (ghunna `ںںں`→`ں` twice, one madd `ا`), not low-confidence phonemes, so posterior scoring should be checked on whether it recovers held-sound length. Ghunna-length cost was done in 04 (875 now 0.25); confidence-from-cost-only was tried and rejected, see 04.
- Implemented (2026-10-05): CTC scoring in `packages/core/src/ctc.ts`; reranking and validation in `createTracker`; scoring rules in `docs/evidence-contract.md`. Tests in `test/posteriors.test.ts`. Regression snapshots now replay decoded-only and are byte-identical to before.
- Mangled-decode test: 30:38 and 30:39 share a 31-phoneme window differing only by و/ف. A decode with و locks 30:38; with posteriors favouring ف, 30:39 ranks first (ambiguous, lead under one edit).
- Comparison: `pnpm --filter @tilawa/core replay --compare-posteriors test/fixtures/tlog/*.json`. 6 of 17 differ, no state changes; 875 confidence 0.25 → 0.33, 680 top moves to its filename ayah 100:5, 318/450 tops move to other surahs (still ambiguous).
- Shard 0 via `relabel --posteriors`: locked 696 → 703, agree 464 → 468, no locked span changed. Ambiguous → locked 7 (incl. 87 → 100:11, filename ayah; 11 → 4:39 for filename 100:10, unverified), ambiguous → insufficient 9 (single-word spans), insufficient → locked 1, locked → insufficient 1 (359, a repeated 100:2 that had locked as 100:1–100:2). Run time 47 s → 67 s per shard.
- Lessons: scoring each candidate only on the frames it aligned made candidates incomparable (87 went to 83:15); scoring over all frames fixed that. Raw CTC charged held-sound length in full (`۾۾۾` vs decoded `م`), so held runs collapse before scoring and fit takes min(decoded, posterior). `NATS_PER_EDIT` 3 is untuned.
- `fuzzy.ts` ghunna-length costs treat `ں` as a ghunna run but not `۾`; worth checking against the 04 ghunna rule.
