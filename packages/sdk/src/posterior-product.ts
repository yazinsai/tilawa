import type { LoadedModel, Tensor } from "./model.js";

/**
 * An in-memory pair of streaming CTC models with a fixed, equal-weight logarithmic pool.
 * The listener supplies one fbank window to both backends and owns both sets of state,
 * so sessions and cold crops remain independent even when they share this model.
 * Pool the full distributions before top-k truncation: q(c) ∝ sqrt(p0(c) p1(c)).
 */
export function posteriorProductModel(first: LoadedModel, second: LoadedModel): LoadedModel {
  const models = [first, second];
  const a = first.manifest, b = second.manifest;
  for (const key of ["inventory", "tokens", "blank", "sampleRate", "frameMs", "window", "shift"] as const) {
    if (a[key] !== b[key]) throw new Error(`Posterior product requires matching ${key}`);
  }
  if (a.fbank.mels !== b.fbank.mels || a.fbank.padValue !== b.fbank.padValue)
    throw new Error("Posterior product requires matching fbank parameters");
  const name = (ear: number, state: string) => `ear${ear}_${state}`;
  return {
    // This manifest describes the in-memory backend; it is not a disk export.
    manifest: {
      ...a,
      id: `${a.id}+${b.id}/posterior-product`,
      states: models.flatMap((model, ear) => model.manifest.states.map(s => ({ ...s, name: name(ear, s.name) }))),
    },
    supported: first.supported.filter(c => second.supported.includes(c)),
    backend: {
      name: `posterior-product(${first.backend.name},${second.backend.name})`,
      async run(feeds) {
        const outputs = await Promise.all(models.map((model, ear) => model.backend.run({
          x: feeds.x,
          ...Object.fromEntries(model.manifest.states.map(s => [s.name, feeds[name(ear, s.name)]])),
        })));
        const left = outputs[0].log_probs, right = outputs[1].log_probs;
        if (left.type !== "float32" || right.type !== "float32" || left.dims.length !== right.dims.length ||
            left.dims.some((d, i) => d !== right.dims[i]) || left.data.length !== right.data.length ||
            left.dims.at(-1) !== a.tokens.length + 1)
          throw new Error("Posterior product requires matching log-probability frames");
        const lp = new Float32Array(left.data.length);
        const vocab = a.tokens.length + 1;
        for (let row = 0; row < lp.length; row += vocab) {
          let max = -Infinity;
          for (let c = 0; c < vocab; c++) {
            const i = row + c;
            lp[i] = (Number(left.data[i]) + Number(right.data[i])) / 2;
            max = Math.max(max, lp[i]);
          }
          if (!Number.isFinite(max)) throw new Error("Posterior product has no shared probability mass");
          let sum = 0;
          for (let c = 0; c < vocab; c++) sum += Math.exp(lp[row + c] - max);
          const logSum = max + Math.log(sum);
          for (let c = 0; c < vocab; c++) lp[row + c] -= logSum;
        }
        const result: Record<string, Tensor> = { log_probs: { type: "float32", data: lp, dims: left.dims } };
        models.forEach((model, ear) => model.manifest.states.forEach(s => {
          result[`new_${name(ear, s.name)}`] = outputs[ear][`new_${s.name}`];
        }));
        return result;
      },
      async release() {
        const results = await Promise.allSettled(models.map(model => model.backend.release()));
        for (const result of results) if (result.status === "rejected") throw result.reason;
      },
    },
  };
}
