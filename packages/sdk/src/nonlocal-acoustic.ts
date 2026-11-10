import { fft512 } from "./fbank.js";

const WIN = 400, HOP = 160, FFT = 512, BINS = 257, BANDS = 16;
const window = Float64Array.from({ length: WIN }, (_, i) => Math.sin(Math.PI * (i + 0.5) / WIN));
const edges = Array.from({ length: BANDS + 1 }, (_, i) => Math.round(2 * Math.pow(128, i / BANDS)));
const FLOOR = 1e-20;

export interface NonlocalGeometry {
  radius: number;
  contextLimit: number;
  consistencyLimit: number;
  donors: number;
}

/** Preregistered candidate, used by algorithm tests and the training-only search. */
export const NONLOCAL_CANDIDATE: NonlocalGeometry = {
  radius: 5, contextLimit: 0.5, consistencyLimit: 0.5, donors: 3,
};
/** No training profile has been validated yet. Populate from training-only calibration before deployment. */
export const NONLOCAL_GEOMETRY: NonlocalGeometry | null = null;

export interface AcousticDonor {
  center: number;
  fromSample: number;
  toSample: number;
  distance: number;
  gain: number;
}
export interface AcousticRepair {
  samples: Float32Array;
  patches: { center: number; donors: AcousticDonor[] }[];
}

function spectrum(audio: Float32Array, offset: number) {
  const re = new Float64Array(FFT), im = new Float64Array(FFT);
  for (let i = 0; i < WIN; i++) re[i] = audio[offset + i] * window[i];
  fft512(re, im);
  return { re, im };
}
function bands(power: ArrayLike<number>, scale = 1): Float32Array {
  return Float32Array.from({ length: BANDS }, (_, b) => {
    let sum = 0;
    for (let k = edges[b]; k < edges[b+1]; k++) sum += power[k];
    return Math.log(Math.max(FLOOR, sum * scale / (edges[b+1] - edges[b])));
  });
}
function distance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return sum / a.length;
}

/** A recording-local bank; retrieval takes no text, phoneme identities or model outputs. */
export class NonlocalAcousticDenoiser {
  readonly power: Float32Array[] = [];
  private readonly logs: Float32Array[] = [];
  private readonly energy: number[] = [];
  private readonly contexts = new Map<number, { context: Float32Array; gain: number } | null>();
  constructor(private readonly audio: Float32Array, readonly geometry = NONLOCAL_CANDIDATE) {
    if (!Number.isInteger(geometry.radius) || geometry.radius < 4 || !Number.isInteger(geometry.donors) || geometry.donors < 3 ||
      !Number.isFinite(geometry.contextLimit) || geometry.contextLimit <= 0 ||
      !Number.isFinite(geometry.consistencyLimit) || geometry.consistencyLimit <= 0)
      throw new RangeError("Invalid nonlocal acoustic geometry");
    for (let at = 0; at + WIN <= audio.length; at += HOP) {
      const { re, im } = spectrum(audio, at);
      const p = Float32Array.from({ length: BINS }, (_, k) => re[k] ** 2 + im[k] ** 2);
      this.power.push(p); this.logs.push(bands(p));
      this.energy.push(p.reduce((a,b) => a+b, 0));
    }
  }
  private extent(t: number) {
    return { fromSample: (t-this.geometry.radius)*HOP, toSample: (t+this.geometry.radius)*HOP+WIN };
  }
  private descriptor(t: number): { context: Float32Array; gain: number } | null {
    if (this.contexts.has(t)) return this.contexts.get(t)!;
    const r = this.geometry.radius;
    if (t < r || t+r >= this.power.length) return null;
    // The inner flank windows share no samples with the central 25 ms window.
    const offsets = [-r, -3, 3, r];
    const gain = offsets.reduce((a,o) => a+this.energy[t+o],0) / offsets.length;
    if (gain <= FLOOR) return null;
    const level = Math.log(gain);
    const result = { gain, context: Float32Array.from(offsets.flatMap(o => Array.from(this.logs[t+o], x => x-level))) };
    this.contexts.set(t,result);
    return result;
  }
  /** All donors end before the disputed word and do not share any samples with each other or the query patch. */
  retrieve(t: number, disputedFrom: number, disputedTo: number): AcousticDonor[] {
    const query = this.descriptor(t);
    if (!query || disputedTo <= disputedFrom) return [];
    const candidates: AcousticDonor[] = [], r = this.geometry.radius, extent = this.extent(t);
    for (let at = r; at+r < this.power.length; at++) {
      const e = this.extent(at);
      if (e.toSample > disputedFrom || e.toSample > extent.fromSample) break;
      const donor = this.descriptor(at);
      if (!donor) continue;
      const d = distance(query.context, donor.context);
      if (d > this.geometry.contextLimit) continue;
      const c = { center: at, ...e, distance: d, gain: query.gain / donor.gain };
      // Retain a bounded shortlist; the center is never used to rank a donor.
      let j = candidates.findIndex(x => x.distance > d);
      if (j < 0) j = candidates.length;
      candidates.splice(j,0,c);
      if (candidates.length > 64) candidates.pop();
    }
    // Try every seed rather than making the nearest inconsistent patch dictate abstention.
    for (let seed = 0; seed < candidates.length; seed++) {
      const chosen: AcousticDonor[] = [];
      for (const c of [candidates[seed], ...candidates.filter((_,i) => i !== seed)]) {
        if (chosen.some(x => c.fromSample < x.toSample && x.fromSample < c.toSample)) continue;
        const center = bands(this.power[c.center], c.gain);
        if (chosen.some(x => distance(center, bands(this.power[x.center],x.gain)) > this.geometry.consistencyLimit)) continue;
        chosen.push(c);
        if (chosen.length === this.geometry.donors) return chosen;
      }
    }
    return [];
  }
  /** Central power averages with original phase. Delta overlap-add keeps unmodified samples exact. */
  repair(fromSample: number, toSample: number): AcousticRepair {
    if (!Number.isInteger(fromSample) || !Number.isInteger(toSample) || fromSample < 0 ||
      toSample > this.audio.length || toSample <= fromSample) throw new RangeError("Invalid acoustic repair span");
    const delta = new Float64Array(toSample-fromSample), norm = new Float64Array(delta.length);
    const patches: AcousticRepair["patches"] = [];
    for (let t = Math.max(0, Math.floor((fromSample-WIN)/HOP)+1); t < this.power.length && t*HOP < toSample; t++) {
      const ds = this.retrieve(t,fromSample,toSample);
      const { re, im } = spectrum(this.audio,t*HOP);
      if (ds.length) {
        patches.push({ center:t, donors:ds });
        const weights = ds.map(d => Math.exp(-d.distance/this.geometry.contextLimit));
        const total = 1 + weights.reduce((a,b) => a+b,0);
        for (let k = 0; k < BINS; k++) {
          const original = this.power[t][k];
          const average = (original + ds.reduce((a,d,i) => a+weights[i]*this.power[d.center][k]*d.gain,0))/total;
          const change = original > FLOOR ? Math.sqrt(average/original)-1 : 0;
          re[k] *= change; im[k] *= change;
          if (k > 0 && k < FFT/2) { re[FFT-k] = re[k]; im[FFT-k] = -im[k]; }
        }
        // Inverse FFT via conjugation; only the real delta is needed.
        for (let k = 0; k < FFT; k++) im[k] = -im[k];
        fft512(re,im);
      }
      for (let i = 0; i < WIN; i++) {
        const at = t*HOP+i-fromSample;
        if (at < 0 || at >= delta.length) continue;
        norm[at] += window[i] ** 2;
        if (ds.length) delta[at] += re[i]/FFT*window[i];
      }
    }
    const samples = this.audio.slice();
    for (let i = 0; i < delta.length; i++) if (norm[i] > FLOOR) samples[fromSample+i] += delta[i]/norm[i];
    return { samples, patches };
  }
}
