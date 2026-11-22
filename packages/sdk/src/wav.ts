/** PCM WAV (8/16/24/32-bit int or 32-bit float) to mono float samples in [-1, 1]. */
export function decodeWav(bytes: Uint8Array): { sampleRate: number; samples: Float32Array } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (o: number) => String.fromCharCode(...bytes.subarray(o, o + 4));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("Not a RIFF/WAVE file");
  let format = 0, channels = 0, sampleRate = 0, bits = 0;
  for (let o = 12; o + 8 <= bytes.length; ) {
    const id = tag(o);
    const size = view.getUint32(o + 4, true);
    const body = o + 8;
    if (id === "fmt ") {
      format = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
      if (format === 0xfffe) format = view.getUint16(body + 24, true);
    } else if (id === "data") {
      const width = bits / 8;
      const n = Math.floor(Math.min(size, bytes.length - body) / (width * channels));
      const samples = new Float32Array(n);
      const read = (p: number) =>
        format === 3 ? view.getFloat32(p, true)
        : bits === 16 ? view.getInt16(p, true) / 32768
        : bits === 32 ? view.getInt32(p, true) / 2147483648
        : bits === 24 ? ((view.getUint8(p) | (view.getUint8(p + 1) << 8) | (view.getInt8(p + 2) << 16)) / 8388608)
        : (view.getUint8(p) - 128) / 128;
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < channels; c++) s += read(body + (i * channels + c) * width);
        samples[i] = s / channels;
      }
      return { sampleRate, samples };
    }
    o = body + size + (size & 1);
  }
  throw new Error("WAV has no data chunk");
}
