/**
 * Verse/span prefix trie for constrained CTC beam search.
 *
 * Each root-to-leaf path represents the phoneme-token sequence of a Quran verse
 * (or a multi-verse span of 2-3 consecutive verses). During beam search every
 * hypothesis is guaranteed to be a prefix of some real verse/span.
 *
 * Uses a two-phase build: first insert into a temporary Map-based tree,
 * then compact into flat typed arrays for minimal memory.
 */

export interface VerseRef {
  /** 0-based index into the quran_phonemes array */
  verseIndex: number;
  /** Number of verses in this span (1 for single verse, 2-3 for spans) */
  spanLength: number;
}

/** Number of phoneme tokens (0-68, excluding blank=69) */
const NUM_TOKENS = 69;

// ── Temporary build-phase node ──────────────────────────────────────────

interface BuildNode {
  children: Map<number, BuildNode>;
  verseEnds: VerseRef[];
  /** Assigned compact index (filled during compaction) */
  compactIdx: number;
}

function createBuildNode(): BuildNode {
  return { children: new Map(), verseEnds: [], compactIdx: -1 };
}

function insertBuild(root: BuildNode, tokenIds: number[], ref: VerseRef): void {
  let node = root;
  for (const id of tokenIds) {
    let child = node.children.get(id);
    if (!child) {
      child = createBuildNode();
      node.children.set(id, child);
    }
    node = child;
  }
  node.verseEnds.push(ref);
}

// ── Compact trie ────────────────────────────────────────────────────────

/**
 * Compact trie using flat typed arrays with sparse child storage.
 *
 * Each node owns a contiguous slice of the edges arrays:
 *   edgeToken[edgeStart[i] .. edgeStart[i]+edgeCount[i]]  = child token IDs (sorted)
 *   edgeChild[edgeStart[i] .. edgeStart[i]+edgeCount[i]]  = child node indices
 *
 * This uses ~6 bytes per edge (Uint8 token + Int32 child) plus ~6 bytes per node
 * (Int32 edgeStart + Uint8 edgeCount). With ~1.7M nodes and ~1.7M edges this is ~20MB.
 */
export class CompactTrie {
  /** Per-node: offset into edge arrays */
  edgeStart: Uint32Array;
  /** Per-node: number of children */
  edgeCount: Uint8Array;
  /** Edge token IDs (sorted per node) */
  edgeToken: Uint8Array;
  /** Edge child node indices */
  edgeChild: Int32Array;
  /** Per-node: verse end refs (empty array if none) */
  verseEnds: VerseRef[][];
  /** Total number of nodes */
  nodeCount: number;

  constructor(
    edgeStart: Uint32Array,
    edgeCount: Uint8Array,
    edgeToken: Uint8Array,
    edgeChild: Int32Array,
    verseEnds: VerseRef[][],
    nodeCount: number,
  ) {
    this.edgeStart = edgeStart;
    this.edgeCount = edgeCount;
    this.edgeToken = edgeToken;
    this.edgeChild = edgeChild;
    this.verseEnds = verseEnds;
    this.nodeCount = nodeCount;
  }

  /**
   * Get the child node index for (nodeIndex, tokenId), or -1 if none.
   * Uses linear scan (most nodes have 1-5 children, so this is fast).
   */
  getChild(nodeIndex: number, tokenId: number): number {
    const start = this.edgeStart[nodeIndex];
    const count = this.edgeCount[nodeIndex];
    for (let i = 0; i < count; i++) {
      const t = this.edgeToken[start + i];
      if (t === tokenId) return this.edgeChild[start + i];
      if (t > tokenId) return -1; // sorted, so no need to continue
    }
    return -1;
  }

  /**
   * Walk from a node following a token sequence.
   * Returns the final node index, or -1 if invalid.
   */
  walk(startNode: number, tokenIds: number[]): number {
    let node = startNode;
    for (const id of tokenIds) {
      node = this.getChild(node, id);
      if (node === -1) return -1;
    }
    return node;
  }

  /**
   * Get all valid child token IDs for a node.
   */
  validChildren(nodeIndex: number): number[] {
    const start = this.edgeStart[nodeIndex];
    const count = this.edgeCount[nodeIndex];
    const result: number[] = new Array(count);
    for (let i = 0; i < count; i++) {
      result[i] = this.edgeToken[start + i];
    }
    return result;
  }

  /**
   * Iterate over children of a node. Calls fn(tokenId, childNodeIndex) for each.
   */
  forEachChild(nodeIndex: number, fn: (tokenId: number, childIdx: number) => void): void {
    const start = this.edgeStart[nodeIndex];
    const count = this.edgeCount[nodeIndex];
    for (let i = 0; i < count; i++) {
      fn(this.edgeToken[start + i], this.edgeChild[start + i]);
    }
  }

  /**
   * Collect all verse ends reachable from a given node (including the node itself).
   */
  collectVerseEnds(nodeIndex: number): VerseRef[] {
    const results: VerseRef[] = [];
    const stack: number[] = [nodeIndex];
    while (stack.length > 0) {
      const n = stack.pop()!;
      for (const ref of this.verseEnds[n]) results.push(ref);
      const start = this.edgeStart[n];
      const count = this.edgeCount[n];
      for (let i = 0; i < count; i++) {
        stack.push(this.edgeChild[start + i]);
      }
    }
    return results;
  }

  /**
   * Estimate memory usage in bytes (typed arrays only, excludes JS overhead).
   */
  estimateMemoryBytes(): number {
    return (
      this.edgeStart.byteLength +
      this.edgeCount.byteLength +
      this.edgeToken.byteLength +
      this.edgeChild.byteLength
      // verseEnds arrays are negligible (most are empty)
    );
  }
}

// ── Compaction ───────────────────────────────────────────────────────────

function compact(root: BuildNode): CompactTrie {
  // BFS to assign indices and count edges
  let nodeCount = 0;
  let totalEdges = 0;
  const queue: BuildNode[] = [root];
  const ordered: BuildNode[] = [];

  while (queue.length > 0) {
    const node = queue.shift()!;
    node.compactIdx = nodeCount++;
    ordered.push(node);
    totalEdges += node.children.size;
    // Sort children by token ID for deterministic ordering
    const sortedKeys = [...node.children.keys()].sort((a, b) => a - b);
    for (const k of sortedKeys) {
      queue.push(node.children.get(k)!);
    }
  }

  const edgeStart = new Uint32Array(nodeCount);
  const edgeCount = new Uint8Array(nodeCount);
  const edgeToken = new Uint8Array(totalEdges);
  const edgeChild = new Int32Array(totalEdges);
  const verseEnds: VerseRef[][] = new Array(nodeCount);

  let edgeOffset = 0;
  for (const node of ordered) {
    const idx = node.compactIdx;
    edgeStart[idx] = edgeOffset;
    const sortedKeys = [...node.children.keys()].sort((a, b) => a - b);
    edgeCount[idx] = sortedKeys.length;
    verseEnds[idx] = node.verseEnds;

    for (const tokenId of sortedKeys) {
      edgeToken[edgeOffset] = tokenId;
      edgeChild[edgeOffset] = node.children.get(tokenId)!.compactIdx;
      edgeOffset++;
    }
  }

  return new CompactTrie(edgeStart, edgeCount, edgeToken, edgeChild, verseEnds, nodeCount);
}

// ── Public API ──────────────────────────────────────────────────────────

export interface QuranVerse {
  surah: number;
  ayah: number;
  phonemes: string;
  [key: string]: unknown;
}

function buildTokenMap(vocabJson: Record<string, string>): Map<string, number> {
  const map = new Map<string, number>();
  for (const [id, token] of Object.entries(vocabJson)) {
    if (token !== "<blank>") {
      map.set(token, parseInt(id));
    }
  }
  return map;
}

function phonemesToTokenIds(
  phonemes: string,
  tokenMap: Map<string, number>,
): number[] {
  const ids: number[] = [];
  for (const tok of phonemes.trim().split(/\s+/)) {
    if (!tok) continue;
    const id = tokenMap.get(tok);
    if (id !== undefined) ids.push(id);
  }
  return ids;
}

export interface TrieStats {
  nodeCount: number;
  totalEdges: number;
  singleVerseCount: number;
  spanCount: number;
  maxDepth: number;
  memoryMB: number;
}

export interface BuiltTrie {
  trie: CompactTrie;
  stats: TrieStats;
}

/**
 * Build the verse/span prefix trie from quran_phonemes data.
 *
 * @param verses - Array of verse objects (from quran_phonemes.json)
 * @param vocabJson - Token vocab mapping (from phoneme_vocab.json)
 * @param maxSpanLength - Maximum consecutive verses to concatenate (default 3)
 */
export function buildTrie(
  verses: QuranVerse[],
  vocabJson: Record<string, string>,
  maxSpanLength = 3,
): BuiltTrie {
  const tokenMap = buildTokenMap(vocabJson);
  const root = createBuildNode();

  const verseTokenIds: number[][] = verses.map((v) =>
    phonemesToTokenIds(v.phonemes, tokenMap),
  );

  let singleVerseCount = 0;
  let spanCount = 0;
  let maxDepth = 0;

  // Insert single verses
  for (let i = 0; i < verses.length; i++) {
    const ids = verseTokenIds[i];
    if (ids.length > 0) {
      insertBuild(root, ids, { verseIndex: i, spanLength: 1 });
      singleVerseCount++;
      if (ids.length > maxDepth) maxDepth = ids.length;
    }
  }

  // Insert multi-verse spans within the same surah
  const WORD_BOUNDARY_ID = tokenMap.get("|")!;

  for (let spanLen = 2; spanLen <= maxSpanLength; spanLen++) {
    for (let start = 0; start + spanLen - 1 < verses.length; start++) {
      if (verses[start].surah !== verses[start + spanLen - 1].surah) continue;

      const combined: number[] = [];
      let valid = true;
      for (let j = 0; j < spanLen; j++) {
        const ids = verseTokenIds[start + j];
        if (ids.length === 0) { valid = false; break; }
        if (j > 0) combined.push(WORD_BOUNDARY_ID);
        for (const id of ids) combined.push(id);
      }

      if (valid && combined.length > 0) {
        insertBuild(root, combined, { verseIndex: start, spanLength: spanLen });
        spanCount++;
        if (combined.length > maxDepth) maxDepth = combined.length;
      }
    }
  }

  // Compact into typed arrays
  const trie = compact(root);

  const memoryBytes = trie.estimateMemoryBytes();

  return {
    trie,
    stats: {
      nodeCount: trie.nodeCount,
      totalEdges: trie.edgeToken.length,
      singleVerseCount,
      spanCount,
      maxDepth,
      memoryMB: memoryBytes / (1024 * 1024),
    },
  };
}
