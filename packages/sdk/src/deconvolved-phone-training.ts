/** Independent, timing-preserving additive observation errors. No per-view
 * filtering: shared native recording noise remains part of the latent occurrence.
 * The SNR and coloration are fixed before practice inspection. */
export function degradedReplicate(samples: Float32Array, seed: number): Float32Array {
  if (!Number.isSafeInteger(seed) || seed <= 0) throw new Error("Invalid replicate seed");
  let state=seed>>>0;
  const random=()=> {state=(Math.imul(state,1664525)+1013904223)>>>0;return (state+.5)/4294967296;};
  const noise=new Float32Array(samples.length);
  let colored=0,power=0,noisePower=0;
  for(let i=0;i<samples.length;i++) {
    colored=.65*colored+.35*(random()+random()+random()-1.5);
    noise[i]=colored;power+=samples[i]*samples[i];noisePower+=colored*colored;
  }
  const level=noisePower>0?Math.sqrt(power/noisePower)*.1:0; // 20 dB SNR
  return samples.map((v,i)=>v+level*noise[i]);
}
