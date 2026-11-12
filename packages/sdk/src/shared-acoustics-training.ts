/** Training-only independent channel/noise replicates. No resampling, cropping or
 * time warping: cross-view covariance compares exactly the same acoustic clock.
 * Seeds and ranges are fixed before inspecting practice comparisons. */
export function degradedReplicate(samples: Float32Array, seed: number): Float32Array {
  if (!Number.isSafeInteger(seed) || seed <= 0) throw new Error("Invalid replicate seed");
  let state=seed>>>0;
  const random=()=> {state=(Math.imul(state,1664525)+1013904223)>>>0;return (state+.5)/4294967296;};
  const high=80+120*random(), low=3500+3000*random(), snr=15+15*random();
  const hp=Math.exp(-2*Math.PI*high/16000), lp=1-Math.exp(-2*Math.PI*low/16000);
  const delay=Math.round(160+480*random()), echo=.04+.12*random();
  const signal=new Float32Array(samples.length), noise=new Float32Array(samples.length);
  let previous=0, highpass=0, lowpass=0, colored=0, power=0, noisePower=0;
  for (let i=0;i<samples.length;i++) {
    highpass=hp*(highpass+samples[i]-previous);previous=samples[i];
    lowpass+=lp*(highpass-lowpass);
    signal[i]=lowpass+(i>=delay?echo*signal[i-delay]:0);
    colored=.65*colored+.35*(random()+random()+random()-1.5);
    noise[i]=colored;power+=signal[i]*signal[i];noisePower+=colored*colored;
  }
  const level=noisePower>0?Math.sqrt(power/noisePower)*Math.pow(10,-snr/20):0;
  for (let i=0;i<signal.length;i++) signal[i]+=level*noise[i];
  return signal;
}
