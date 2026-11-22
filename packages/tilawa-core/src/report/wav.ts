/** Canonical 44-byte WAV: PCM16 LE, mono, 16 kHz. The same layout `/live` and `live_review_core.validate` accept. */

export const WAV_HEADER_BYTES = 44;
/** `(fmtSize, audioFormat, channels, sampleRate, byteRate, blockAlign, bitsPerSample)` the validator requires. */
export const WAV_FMT = [16, 1, 1, 16000, 32000, 2, 16] as const;

export function encodeWav(pcm: Int16Array): Uint8Array {
  const dataBytes = pcm.length * 2;
  const out = new Uint8Array(WAV_HEADER_BYTES + dataBytes);
  const view = new DataView(out.buffer);
  const ascii = (at: number, s: string) => {
    for (let i = 0; i < s.length; i++) out[at + i] = s.charCodeAt(i);
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, WAV_FMT[0], true);
  view.setUint16(20, WAV_FMT[1], true);
  view.setUint16(22, WAV_FMT[2], true);
  view.setUint32(24, WAV_FMT[3], true);
  view.setUint32(28, WAV_FMT[4], true);
  view.setUint16(32, WAV_FMT[5], true);
  view.setUint16(34, WAV_FMT[6], true);
  ascii(36, "data");
  view.setUint32(40, dataBytes, true);
  for (let i = 0; i < pcm.length; i++) view.setInt16(WAV_HEADER_BYTES + 2 * i, pcm[i], true);
  return out;
}

/** Returns the problems with a WAV against the canonical layout; empty when it is canonical. */
export function wavProblems(wav: Uint8Array): string[] {
  if (wav.length < WAV_HEADER_BYTES) return ["audio.wav is shorter than its 44-byte header"];
  const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  const tag = (at: number) => String.fromCharCode(wav[at], wav[at + 1], wav[at + 2], wav[at + 3]);
  const errors: string[] = [];
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") errors.push("audio.wav is not RIFF/WAVE");
  if (tag(12) !== "fmt " || tag(36) !== "data") errors.push("audio.wav does not have the canonical 44-byte header");
  const fmt = [
    view.getUint32(16, true),
    view.getUint16(20, true),
    view.getUint16(22, true),
    view.getUint32(24, true),
    view.getUint32(28, true),
    view.getUint16(32, true),
    view.getUint16(34, true),
  ];
  if (fmt.some((v, i) => v !== WAV_FMT[i]))
    errors.push(`audio.wav fmt is (${fmt.join(",")}), expected (${WAV_FMT.join(",")}) mono 16 kHz PCM16`);
  if (view.getUint32(40, true) !== wav.length - WAV_HEADER_BYTES) errors.push("audio.wav data size does not match the file");
  if (view.getUint32(4, true) !== wav.length - 8) errors.push("audio.wav RIFF size does not match the file");
  if ((wav.length - WAV_HEADER_BYTES) % 2 !== 0) errors.push("audio.wav has an odd number of data bytes");
  return errors;
}

/** PCM16 samples of a canonical WAV (call `wavProblems` first). */
export function wavSamples(wav: Uint8Array): Int16Array {
  const n = Math.floor((wav.length - WAV_HEADER_BYTES) / 2);
  const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  const out = new Int16Array(n);
  for (let i = 0; i < n; i++) out[i] = view.getInt16(WAV_HEADER_BYTES + 2 * i, true);
  return out;
}
