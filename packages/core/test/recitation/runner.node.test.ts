import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { KaldiFbank } from "../../src/recitation/fbank";
import { GreedyCtcDecoder } from "../../src/recitation/ctcDecoder";
import { ZipformerRunner, type ZipformerIo } from "../../src/recitation/zipformerRunner";
import { TOKENS, BLANK_ID } from "../../src/recitation/tokens";
import { SAMPLE_RATE } from "../../src/recitation/config";
import {
  findClip,
  findLegacyModel,
  findLegacyModelIo,
  findOrtDir,
  findPython,
  findPythonRoot,
  VECTORS,
} from "./paths";

const PY = findPython();
const PY_ROOT = findPythonRoot();
const CLIP = findClip();
// The recorded ctc_001002.json stream comes from the previous default model.
const MODEL = findLegacyModel();
const IO_PATH = findLegacyModelIo();
const ORT_DIR = findOrtDir();
const CHUNK = 7680;
const TAIL_SECONDS = 2.0;

const haveModel = !!(MODEL && IO_PATH && CLIP && PY && PY_ROOT);
const haveOrt = !!ORT_DIR && existsSync(resolve(ORT_DIR, "onnxruntime-node"));

function loadMp3(path: string): Float32Array {
  const dir = mkdtempSync(join(tmpdir(), "ctc-"));
  const out = join(dir, "clip.f32");
  const script = `
import sys
sys.path.insert(0, ${JSON.stringify(PY_ROOT)})
from shared.audio import load_audio
w = load_audio(${JSON.stringify(path)}, sr=16000)
w.astype("float32").tofile(${JSON.stringify(out)})
`;
  execFileSync(PY!, ["-c", script]);
  const buf = readFileSync(out);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

describe.skipIf(!haveModel || !haveOrt)("zipformer runner (onnxruntime-node)", () => {
  it("reproduces ctc_001002.json token stream", async () => {
    const vec = JSON.parse(readFileSync(resolve(VECTORS, "ctc_001002.json"), "utf8")) as {
      framesDecoded: number;
      leftoverFbankFrames: number;
      tokenCount: number;
      transcript: string;
      tokens: { sym: string; frame: number; margin: number }[];
    };
    const require = createRequire(`${ORT_DIR!}/`);
    const ort = require("onnxruntime-node") as unknown;
    const io = JSON.parse(readFileSync(IO_PATH!, "utf8")) as ZipformerIo;
    const runner = await ZipformerRunner.create(
      ort,
      new Uint8Array(readFileSync(MODEL!)),
      io,
      ["cpu"],
    );
    const fbank = new KaldiFbank();
    const decoder = new GreedyCtcDecoder(TOKENS, BLANK_ID);
    const pcm = loadMp3(CLIP!);
    const tokens: { sym: string; frame: number; margin: number }[] = [];

    const consumeFrames = async (frames: Float32Array[]) => {
      if (!frames.length) return;
      const { logProbs, frames: n } = await runner.accept(frames);
      if (n === 0) return;
      tokens.push(...decoder.consume(logProbs, n, runner.io.vocabSize));
    };

    for (let i = 0; i < pcm.length; i += CHUNK) {
      await consumeFrames(fbank.acceptWaveform(pcm.subarray(i, Math.min(pcm.length, i + CHUNK))));
    }
    const silence = new Float32Array(Math.round(TAIL_SECONDS * SAMPLE_RATE));
    await consumeFrames(fbank.acceptWaveform(silence));
    await consumeFrames(fbank.inputFinished());
    tokens.push(...decoder.flush());

    expect(decoder.framesDecoded).toBe(vec.framesDecoded);
    expect(runner.leftoverFrames).toBe(vec.leftoverFbankFrames);
    expect(tokens.length).toBe(vec.tokenCount);
    expect(tokens.map((t) => t.sym).join("")).toBe(vec.transcript);
    expect(tokens.map((t) => t.sym)).toEqual(vec.tokens.map((t) => t.sym));
    expect(tokens.map((t) => t.frame)).toEqual(vec.tokens.map((t) => t.frame));
    // 1e-4: int8 EP softmax margin jitter vs the dump machine.
    for (let i = 0; i < tokens.length; i++) {
      expect(Math.abs(tokens[i]!.margin - vec.tokens[i]!.margin), `margin[${i}]`).toBeLessThan(1e-4);
    }
  }, 180_000);
});
