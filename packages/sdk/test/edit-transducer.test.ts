import { describe, expect, it } from "vitest";
import { decodeEdits, editSource } from "../src/edit-transducer.js";
import { createListener } from "../src/listener.js";
import type { LoadedModel } from "../src/model.js";

function scores(frames: number, vocabulary = 3) {
  const actions = new Float32Array(frames * (frames + 1) * 5).fill(-12);
  const phones = new Float32Array(frames * (frames + 1) * vocabulary).fill(-12);
  return {
    actions, phones,
    action(t: number, i: number, a: number, score = 12) { actions[(t * (frames + 1) + i) * 5 + a] = score; },
    phone(t: number, i: number, p: number) { phones[(t * (frames + 1) + i) * vocabulary + p - 1] = 12; },
  };
}

describe("monotone acoustic edit decoding", () => {
  it("inserts a missing event in blank frames between copied source phones, on the existing clock", () => {
    const s = scores(3);
    s.action(0, 0, 2); s.action(1, 1, 4); s.phone(1, 1, 3); s.action(2, 1, 2);
    const got = decodeEdits(s.actions, s.phones, editSource([1, 0, 2]), 3, "abc", 40);
    expect(got.map(p => [p.p, p.start, p.end])).toEqual([["a", 40, 40], ["c", 41, 41], ["b", 42, 42]]);
  });

  it("deletes a fragment without moving time and substitutes at the source interval", () => {
    const s = scores(3);
    s.action(0, 0, 2); s.action(1, 1, 1); s.action(1, 2, 0);
    s.action(2, 2, 3); s.phone(2, 2, 1);
    const got = decodeEdits(s.actions, s.phones, editSource([1, 2, 3]), 3, "abc");
    expect(got.map(p => [p.p, p.start])).toEqual([["a", 0], ["a", 2]]);
  });

  it("consumes a sustained source run once and preserves blank-separated repetitions", () => {
    expect(editSource([1, 1, 0, 1, 1])).toEqual([{ id: 1, start: 0, end: 1 }, { id: 1, start: 3, end: 4 }]);
    expect(editSource([1, 1, 0, 2], 1)).toEqual([{ id: 2, start: 3, end: 3 }]);
    const s = scores(2);
    s.action(0, 0, 2); s.action(1, 1, 0);
    const got = decodeEdits(s.actions, s.phones, editSource([1, 1]), 2, "abc");
    expect(got).toHaveLength(1);
    expect([got[0].start, got[0].end]).toEqual([0, 1]);
  });

  it("uses a complete script instead of the locally best first edit", () => {
    const s = scores(1);
    s.action(0, 0, 1, 1); s.action(0, 0, 2, 0.9);
    // Deleting first leaves a flat payload distribution and a 50/50 hold/insert decision.
    s.action(0, 1, 0, 0); s.action(0, 1, 4, 0);
    expect(decodeEdits(s.actions, s.phones, editSource([1]), 1, "abc").map(p => p.p)).toEqual(["a"]);
  });

  it("does not let drain padding emit or constrain a valid earlier script", () => {
    const s = scores(3);
    s.action(0, 0, 2); s.action(1, 1, 4); s.phone(1, 1, 3); s.action(2, 1, 2);
    const got = decodeEdits(s.actions, s.phones, editSource([1]), 3, "abc", 0, 1);
    expect(got.map(p => p.p)).toEqual(["a"]);
  });

  it("rejects invalid clocks and nonfinite scores", () => {
    const s = scores(1);
    expect(() => decodeEdits(s.actions, s.phones, [{ id: 1, start: 0, end: 1 }], 1, "abc")).toThrow();
    s.actions[0] = NaN;
    expect(() => decodeEdits(s.actions, s.phones, [], 1, "abc")).toThrow();
  });

  it("uses edit tokens as primary evidence and the independent CTC head as support", async () => {
    const s = scores(2);
    s.action(0, 0, 4); s.phone(0, 0, 2); s.action(1, 0, 0);
    const model: LoadedModel = {
      manifest: { schema: 1, id: "synthetic", format: "onnx", arch: "zipformer2-streaming-ctc",
        inventory: "test", tokens: "abc", blank: 0, sampleRate: 16000, frameMs: 40,
        fbank: { mels: 80, padValue: -23.0259 }, window: 21, shift: 8, states: [],
        file: "unused", bytes: 0, sha256: "unused",
        editDecoder: { kind: "acoustic-edit", actions: "edit_actions", phones: "edit_phones", source: "source_log_probs" } },
      supported: [],
      backend: { name: "stub", async release() {}, async run() {
        const blank = new Float32Array([0, -20, -20, -20, 0, -20, -20, -20]);
        // Support favours c, while the primary edit emits b and the source is blank.
        const support = new Float32Array([-20, -20, -20, 0, 0, -20, -20, -20]);
        return {
          log_probs: { type: "float32", data: support, dims: [1, 2, 4] },
          source_log_probs: { type: "float32", data: blank, dims: [1, 2, 4] },
          edit_actions: { type: "float32", data: s.actions, dims: [1, 2, 3, 5] },
          edit_phones: { type: "float32", data: s.phones, dims: [1, 2, 3, 3] },
        };
      } },
    };
    const listener = createListener(model, { posteriors: true, topK: 2 });
    const e = await listener.push(new Float32Array(21 * 160 + 400), 16000);
    expect(e!.phonemes.map(p => p.p)).toEqual(["b"]);
    expect(Array.from(e!.posteriors!.ids).slice(0, 2)).toEqual([3, 0]);
    model.manifest.editDecoder!.phones = "absent";
    await expect(listener.push(new Float32Array(1600), 16000)).rejects.toThrow("Declared acoustic edit outputs");
    model.manifest.editDecoder!.phones = "edit_phones";
    model.manifest.editFrameCountInput = "edit_valid_frames";
    const backendRun = model.backend.run;
    model.backend.run = async feeds => {
      expect(Array.from(feeds.edit_valid_frames.data as BigInt64Array)).toEqual([1n]);
      return backendRun(feeds);
    };
    const short = createListener(model, { posteriors: true, topK: 2 });
    expect(await short.push(new Float32Array(1680), 16000)).toBeNull();
    const final = await short.end();
    expect(final!.posteriors!.nFrames).toBe(1);
    expect(final!.phonemes.map(p => [p.p, p.start, p.end])).toEqual([["b", 0, 0]]);
  });
});
