/** Run after paired export with TILAWA_MASKED_CONVOLUTIONS_DIR pointing to its parent directory. */
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createListener } from "../src/listener.js";
import { loadNodeModel, readWav } from "../src/node.js";
import { MODEL_REGISTRY } from "../src/registry.js";

const root = process.env.TILAWA_MASKED_CONVOLUTIONS_DIR;
const fixture = new URL("./fixtures/tlog-clean-00010-630.wav", import.meta.url).pathname;

async function sha256(path: string) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

describe.runIf(root)("masked and continuation acoustic bundles through the standard SDK", () => {
  for (const arm of ["masked", "control"]) {
    it(`${arm} preserves the inventory, frame clock, push invariance and insertion assets`, async () => {
      const dir = join(root!, arm);
      const model = await loadNodeModel(dir);
      try {
        const { manifest: m } = model;
        expect(m.inventory).toBe("muaalem-v3_2-phonemes");
        expect(m.tokens).toBe(MODEL_REGISTRY.tiers[0].manifest.tokens);
        expect(m.frameMs).toBe(40);
        expect(m.sampleRate).toBe(16000);
        expect(m.blank).toBe(0);
        expect(await sha256(join(dir, m.file))).toBe(m.sha256);
        for (const asset of MODEL_REGISTRY.auxiliary ?? []) {
          expect(await sha256(join(dir, asset.file))).toBe(asset.sha256);
          for (const external of asset.externalData)
            expect(await sha256(join(dir, external.file))).toBe(external.sha256);
        }
        expect(model.insertionTeacher).toBeDefined();
        const parity = JSON.parse(await readFile(join(dir, "export-parity.json"), "utf8"));
        expect(parity.windows).toBe(3);
        const wav = await readWav(fixture);
        async function decode(step: number) {
          const listener = createListener(model, { posteriors: true });
          const chunks = [];
          for (let offset = 0; offset < wav.samples.length; offset += step) {
            const chunk = await listener.push(wav.samples.subarray(offset, offset + step), wav.sampleRate);
            if (chunk) chunks.push(chunk);
          }
          const last = await listener.end();
          if (last) chunks.push(last);
          let frames = 0;
          for (const chunk of chunks) {
            expect(chunk.inventory).toBe(m.inventory);
            expect(chunk.frameMs).toBe(m.frameMs);
            expect(chunk.posteriors!.startFrame).toBe(frames);
            frames += chunk.posteriors!.nFrames;
            expect(chunk.posteriors!.probs.every(Number.isFinite)).toBe(true);
          }
          return { frames, phones: chunks.flatMap(chunk => chunk.phonemes) };
        }
        const small = await decode(320);
        expect(await decode(16000)).toEqual(small);
        const fbankFrames = Math.floor((wav.samples.length - 400) / 160) + 1;
        expect(small.frames).toBe(Math.floor((Math.floor((fbankFrames - 7) / 2) + 1) / 2));
      } finally {
        await model.backend.release();
      }
    }, 180000);
  }
});
