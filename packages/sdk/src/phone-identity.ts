/** Redistribute the shipped speech mass using a separately distilled identity head.
 * The original blank log probability is copied, including at exact silence.
 */
export function conditionalPhoneRows(base: Float32Array, identity: Float32Array, vocab: number): Float32Array {
  if (vocab < 2 || base.length % vocab || identity.length !== base.length / vocab * (vocab - 1))
    throw new Error("Conditional phone head differs from the primary output geometry");
  const out = new Float32Array(base.length);
  for (let t = 0; t < base.length / vocab; t++) {
    const blank = base[t * vocab];
    if (!Number.isFinite(blank) || blank > 0) throw new Error("Invalid primary blank probability");
    const start = t * (vocab - 1);
    let peak = -Infinity;
    for (let i = 0; i < vocab - 1; i++) {
      const v = identity[start + i];
      if (!Number.isFinite(v)) throw new Error("Invalid conditional phone identity");
      peak = Math.max(peak, v);
    }
    let total = 0;
    for (let i = 0; i < vocab - 1; i++) total += Math.exp(identity[start + i] - peak);
    const normalizer = peak + Math.log(total);
    // expm1 retains speech mass when blank is extremely close to one.
    const speech = Math.log(-Math.expm1(blank));
    out[t * vocab] = blank;
    for (let i = 0; i < vocab - 1; i++) out[t * vocab + i + 1] = speech + identity[start + i] - normalizer;
  }
  return out;
}
