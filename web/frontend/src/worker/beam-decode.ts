/**
 * CTC prefix beam search constrained by a verse/span trie.
 *
 * Every hypothesis is a prefix of some real Quran verse (or multi-verse span).
 * At each CTC frame, we only expand to tokens that are valid trie children,
 * ensuring the search space stays tractable.
 */

import type { CompactTrie, VerseRef } from "../lib/phoneme-trie";

export interface BeamCandidate {
  /** Decoded token IDs (no blanks, no duplicates) */
  tokenIds: number[];
  /** Log-probability score (blank path) */
  blankScore: number;
  /** Log-probability score (non-blank path) */
  nonBlankScore: number;
  /** Current node index in the compact trie */
  trieNodeIdx: number;
  /** Completed verse matches encountered along the path */
  matchedVerses: VerseRef[];
}

export interface BeamResult {
  /** Decoded token IDs */
  tokenIds: number[];
  /** Combined acoustic log-probability (log(p_blank + p_nonblank)) */
  score: number;
  /** Verse references matched at the end of this hypothesis */
  matchedVerses: VerseRef[];
  /** Whether this hypothesis ends at a complete verse */
  isComplete: boolean;
}

const NEG_INF = -Infinity;

function logAddExp(a: number, b: number): number {
  if (a === NEG_INF) return b;
  if (b === NEG_INF) return a;
  const hi = Math.max(a, b);
  return hi + Math.log1p(Math.exp(Math.min(a, b) - hi));
}

function totalScore(h: BeamCandidate): number {
  return logAddExp(h.blankScore, h.nonBlankScore);
}

/**
 * CTC prefix beam search constrained by a verse trie.
 *
 * @param logprobs - Flat Float32Array of log-probabilities [timeSteps x vocabSize]
 * @param timeSteps - Number of CTC frames
 * @param vocabSize - Size of the vocabulary (typically 70)
 * @param blankId - ID of the CTC blank token (typically 69)
 * @param trie - Compact trie built from verse phonemes
 * @param beamWidth - Maximum number of hypotheses to keep per frame (default 8)
 * @returns Top beam candidates sorted by score (best first)
 */
export function beamSearchDecode(
  logprobs: Float32Array,
  timeSteps: number,
  vocabSize: number,
  blankId: number,
  trie: CompactTrie,
  beamWidth = 8,
): BeamResult[] {
  // Initialize with empty hypothesis at trie root (node 0)
  let beams: Map<string, BeamCandidate> = new Map();
  beams.set("", {
    tokenIds: [],
    blankScore: 0,
    nonBlankScore: NEG_INF,
    trieNodeIdx: 0,
    matchedVerses: [],
  });

  for (let t = 0; t < timeSteps; t++) {
    const frameOffset = t * vocabSize;
    const nextBeams: Map<string, BeamCandidate> = new Map();

    for (const beam of beams.values()) {
      const prevTotal = totalScore(beam);
      if (prevTotal === NEG_INF) continue;

      const blankLogProb = logprobs[frameOffset + blankId];
      const blankKey = beam.tokenIds.join(",");

      // 1. Extend with blank: stays at same trie node, same token sequence
      const newBlankScore = prevTotal + blankLogProb;
      const existingBlank = nextBeams.get(blankKey);
      if (existingBlank) {
        existingBlank.blankScore = logAddExp(existingBlank.blankScore, newBlankScore);
      } else {
        nextBeams.set(blankKey, {
          tokenIds: beam.tokenIds,
          blankScore: newBlankScore,
          nonBlankScore: NEG_INF,
          trieNodeIdx: beam.trieNodeIdx,
          matchedVerses: beam.matchedVerses,
        });
      }

      // 2. Extend with each valid child token from the trie
      const nodeIdx = beam.trieNodeIdx;
      const edgeStart = trie.edgeStart[nodeIdx];
      const edgeCount = trie.edgeCount[nodeIdx];

      for (let e = 0; e < edgeCount; e++) {
        const tokenId = trie.edgeToken[edgeStart + e];
        const childIdx = trie.edgeChild[edgeStart + e];

        const tokenLogProb = logprobs[frameOffset + tokenId];
        const lastToken = beam.tokenIds.length > 0
          ? beam.tokenIds[beam.tokenIds.length - 1]
          : -1;

        // CTC duplicate handling
        let newNonBlankScore: number;
        if (tokenId === lastToken) {
          // Repeated token: only blank→nonblank transition
          newNonBlankScore = beam.blankScore + tokenLogProb;
        } else {
          newNonBlankScore = prevTotal + tokenLogProb;
        }

        const newKey = blankKey.length > 0
          ? blankKey + "," + tokenId
          : "" + tokenId;

        const childVerseEnds = trie.verseEnds[childIdx];
        const existing = nextBeams.get(newKey);

        if (existing) {
          existing.nonBlankScore = logAddExp(existing.nonBlankScore, newNonBlankScore);
          if (childVerseEnds.length > 0 && existing.matchedVerses.length === 0) {
            existing.matchedVerses = [...beam.matchedVerses, ...childVerseEnds];
          }
        } else {
          const newMatchedVerses = childVerseEnds.length > 0
            ? [...beam.matchedVerses, ...childVerseEnds]
            : beam.matchedVerses;

          nextBeams.set(newKey, {
            tokenIds: [...beam.tokenIds, tokenId],
            blankScore: NEG_INF,
            nonBlankScore: newNonBlankScore,
            trieNodeIdx: childIdx,
            matchedVerses: newMatchedVerses,
          });
        }
      }
    }

    // Prune to beam width
    if (nextBeams.size > beamWidth) {
      const sorted = [...nextBeams.entries()].sort(
        ([, a], [, b]) => totalScore(b) - totalScore(a),
      );
      beams = new Map(sorted.slice(0, beamWidth));
    } else {
      beams = nextBeams;
    }
  }

  // Convert to results
  const results: BeamResult[] = [...beams.values()]
    .map((beam) => ({
      tokenIds: beam.tokenIds,
      score: totalScore(beam),
      matchedVerses: beam.matchedVerses,
      isComplete: trie.verseEnds[beam.trieNodeIdx].length > 0,
    }))
    .sort((a, b) => b.score - a.score);

  return results;
}
