# 01: Reference index and exact-match locate

**What to build:** A whole clip of exact reference phonemes, pushed into a new tracker, locates to the correct surah, ayah and word span, including when it starts mid-ayah. The tracker reports `locked` only when the match is unique. When an exact phrase occurs in several places, every occurrence remains a candidate from the start. This ticket establishes the public types that every later ticket builds on.

- **Index:** a build step produces a versioned reference index asset from a pinned Uthmani text and `quran-transcript` phonemes with a fixed `MoshafAttributes` profile. Every word has stable surah/ayah/word identifiers and a phoneme span. The index also holds an n-gram inverted index for retrieval. Asset metadata records the text version, phonemizer version and profile.
- **Public API:** `createTracker(index, options)` with `push(evidence)` and `end()`, both returning `EngineEvent[]`.
- **Types fixed here:**
  - `PhonemeEvidence`: decoded phonemes with start/end frame and confidence, plus the frame stride in ms.
  - Position candidate: word span, score fields, and a placeholder confidence.
  - Matched span: start and end word reference, which may cross ayahs.
- **Posterior input contract.** This is specified now and used in 05. Optional per-frame top-k entries carry:
  - the frame stride and token inventory id, which must match the index's phoneme inventory;
  - an explicit blank id and blank probability;
  - a rule for the mass that top-k omits: the remaining probability is spread uniformly over the tokens not in the top-k, with a stated floor.
  
  The tracker must reject evidence whose inventory doesn't match the index.

**Blocked by:** None (can start immediately).

**Status:** done

- [x] The index builds reproducibly from pinned inputs, and its metadata records every version used.
- [x] Exact phonemes for a single ayah give a `locked` position with the correct span.
- [x] Exact phonemes starting mid-ayah give the correct starting word.
- [x] Exact phonemes spanning two consecutive ayahs give a span across both.
- [x] An exact phrase that occurs in several places (for example a repeated verse such as 55:13) never yields `locked`. All occurrences are returned as candidates.
- [x] Evidence with a mismatched token inventory is rejected with a clear error.
- [x] The posterior input contract is documented alongside the types.
- [x] Vitest is set up for the core package, and all tests go through the public API only.

## Comments

- Index `ref-v1`: 6,236 ayahs and 77,433 words, 4.7 MB JSON. The n-gram index is built at load time rather than stored in the asset.
- Word boundaries come from aligning each word phonemized in isolation against the continuous ayah, because cross-word assimilation merges words in 66% of ayahs. One word is fully assimilated and has an empty span.
- Install note: `pnpm add` fails through the shell's security wrapper (registry metadata error), so dev dependencies were installed with npm.
