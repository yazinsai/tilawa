// Review clips cut from the take's canonical WAV (44-byte header, PCM16 LE, mono, 16 kHz).
import { encodeWav } from "@tilawa/core/report";

const HEADER = 44;
const PER_MS = 16;

export function sliceWav(wav: Uint8Array, startMs: number, endMs: number): Uint8Array {
  const total = Math.floor((wav.length - HEADER) / 2);
  const a = Math.max(0, Math.min(total, Math.floor(startMs * PER_MS)));
  const b = Math.max(a, Math.min(total, Math.ceil(endMs * PER_MS)));
  const view = new DataView(wav.buffer, wav.byteOffset + HEADER, (wav.length - HEADER) & ~1);
  const pcm = new Int16Array(b - a);
  for (let i = 0; i < pcm.length; i++) pcm[i] = view.getInt16((a + i) * 2, true);
  return encodeWav(pcm);
}

export function wavUrl(wav: Uint8Array): string {
  return URL.createObjectURL(new Blob([wav as unknown as ArrayBuffer], { type: "audio/wav" }));
}
