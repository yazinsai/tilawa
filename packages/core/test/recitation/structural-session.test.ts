/**
 * The session's structural-rule wiring on a scripted decode (no model file):
 * evaluation at pauses, the flag queue, dismiss-to-next, and flags off leaving
 * the message stream untouched.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { QuranCorpus } from "../../src/recitation/corpus";
import { createZipformerSession, type StructuralOptions } from "../../src/recitation/session";
import { encodePhonemes } from "../../src/recitation/posteriors";
import { BLANK_ID, VOCAB_SIZE } from "../../src/recitation/tokens";
import type { OrtSessionLike, TensorLike } from "../../src/recitation/zipformerRunner";
import type { CorrectionIssue } from "../../src/recitation/correction";
import type { WorkerOutbound } from "../../src/types";
import { requireCorpus } from "./paths";

const CHUNK = 7680;
const FRAMES_PER_RUN = 12;
const corpusJson = JSON.parse(readFileSync(requireCorpus(), "utf8"));
const corpus = new QuranCorpus(corpusJson);

class StubTensor implements TensorLike {
  constructor(readonly type: string, readonly data: Float32Array | BigInt64Array | Int32Array, readonly dims: readonly number[]) {}
}
const TensorCtor = StubTensor as unknown as new (type: string, data: Float32Array | BigInt64Array | Int32Array, dims: readonly number[]) => TensorLike;

/** One token every other frame; `null` entries are 2 blank frames. */
class ScriptedOrtSession implements OrtSessionLike {
  private readonly queue: Array<number | null>;
  constructor(ids: Array<number | null>) { this.queue = [...ids]; }
  get pending(): number { return this.queue.length; }
  async run(feeds: Record<string, TensorLike>): Promise<Record<string, TensorLike>> {
    const out: Record<string, TensorLike> = {};
    for (const [name, tensor] of Object.entries(feeds)) if (name !== "x") out[`new_${name}`] = tensor;
    const data = new Float32Array(FRAMES_PER_RUN * VOCAB_SIZE).fill(Math.log(0.001));
    for (let f = 0; f < FRAMES_PER_RUN; f++) {
      const id = f % 2 === 0 ? this.queue.shift() ?? BLANK_ID : BLANK_ID;
      data[f * VOCAB_SIZE + (id ?? BLANK_ID)] = Math.log(0.9);
    }
    out.log_probs = new StubTensor("float32", data, [1, FRAMES_PER_RUN, VOCAB_SIZE]);
    return out;
  }
}

/** Ayahs read with a 0.8 s pause after each. */
function script(surah: number, ayahs: number[]): Array<number | null> {
  const out: Array<number | null> = [];
  for (const a of ayahs) {
    out.push(...encodePhonemes(corpus.ayahPhonemes(surah, a))!);
    out.push(...new Array(10).fill(null));
  }
  return out;
}

/** Run a take in correction mode, dismissing every issue on sight (as the lab harness does). */
async function take(ids: Array<number | null>, structural?: StructuralOptions, expected?: { surah: number; ayah: number; ayahEnd: number }) {
  const ort = new ScriptedOrtSession(ids);
  const session = await createZipformerSession({ session: ort, Tensor: TensorCtor, corpus: corpusJson, structural: structural ?? false });
  session.setMode("correction");
  if (expected) session.setExpected(expected);
  const issues: CorrectionIssue[] = [];
  const all: WorkerOutbound[] = [];
  const collect = (msgs: WorkerOutbound[]): void => {
    for (const m of msgs) {
      all.push(m);
      if (m.type === "correction" && m.state.phase === "error" && m.state.issue) {
        issues.push(m.state.issue);
        collect(session.correct("dismiss"));
      }
    }
  };
  const chunks = Math.ceil((ids.length * 2) / FRAMES_PER_RUN) + 3;
  for (let i = 0; i < chunks; i++) collect(await session.feed(new Float32Array(CHUNK)));
  collect(await session.stop());
  return { issues, all, session };
}

describe("ZipformerSession structural rules", () => {
  it("is on by default, at stop timing", async () => {
    const session = await createZipformerSession({ session: new ScriptedOrtSession([]), Tensor: TensorCtor, corpus: corpusJson });
    const s = session as unknown as { aoRule: unknown; svOn: boolean; structuralLive: boolean };
    expect(s.aoRule).not.toBeNull();
    expect(s.svOn).toBe(true);
    expect(s.structuralLive).toBe(false);
  });

  it("leaves the message stream unchanged when off", async () => {
    const ids = script(87, [1, 2, 4, 5]);
    const a = await take(ids);
    const b = await take(ids, { ayahOrder: false, similarVerse: false });
    const strip = (ms: WorkerOutbound[]) => JSON.stringify(ms.map((m) => (m.type === "debug" ? { ...m, at: 0 } : m)));
    expect(strip(b.all)).toBe(strip(a.all));
    expect(a.issues.every((i) => i.source === undefined)).toBe(true);
  });

  it("raises each skipped ayah once, from the engine or the ayah-order rule", async () => {
    const off = await take(script(87, [1, 2, 4, 5]));
    expect(off.issues.map((i) => `${i.kind} ${i.ayah}`)).toEqual(["possible_skipped_ayah 3"]);
    // Stop timing: the engine raised 87:3 live, so the end-of-take rule adds nothing.
    const stop = await take(script(87, [1, 2, 4, 5]), { ayahOrder: true, similarVerse: true });
    expect(stop.issues).toEqual(off.issues);
    // Pause timing: whichever of the engine and the rule sees the jump first raises it; the other is deduplicated.
    const live = await take(script(87, [1, 2, 4, 5]), { ayahOrder: true, similarVerse: true, timing: "pause" });
    const skips = live.issues.filter((i) => i.kind === "possible_skipped_ayah");
    expect(skips.map((i) => `${i.surah}:${i.ayah}`)).toEqual(["87:3"]);
    expect(skips[0]!.words).toBe(corpus.ayahWordCount(87, 3));
    expect(skips[0]!.wordIndex).toBe(corpus.ayahFirstWord(87, 3));
  });

  it("raises every end-of-take flag, one per correct() call", async () => {
    // 87:3 and 87:6 skipped; the engine only follows 87:1-2 -> 4 live.
    const on = await take(script(87, [1, 2, 4, 5, 7, 8]), { ayahOrder: true });
    const skips = on.issues.filter((i) => i.kind === "possible_skipped_ayah").map((i) => i.ayah);
    expect(skips).toEqual(expect.arrayContaining([3, 6]));
    expect(new Set(skips).size).toBe(skips.length);
  });

  it("adds no structural flag on a clean in-order take", async () => {
    for (const timing of ["stop", "pause"] as const) {
      const { issues } = await take(script(87, [1, 2, 3, 4, 5]), { ayahOrder: true, similarVerse: true, timing });
      expect(issues.filter((i) => i.source)).toEqual([]);
    }
  });

  it("drops its queue and token log on reset()", async () => {
    const { session } = await take(script(87, [1, 2, 4, 5]), { ayahOrder: true });
    session.reset();
    const st = (session as unknown as { st: { tokens: unknown[]; pending: unknown[]; issues: unknown[] } }).st;
    expect(st.tokens).toEqual([]);
    expect(st.pending).toEqual([]);
    expect(st.issues).toEqual([]);
  });
});
