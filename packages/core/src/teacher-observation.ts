import { UNIT } from "./fuzzy.js";

/** One complete input phone run emits a normalized distribution of output fragments, including epsilon. */
export interface RunTransducer {
  phones: string[];
  fragments: string[];
  rows: number[][];
}
export interface TeacherObservationModel {
  referenceToTeacher: RunTransducer;
  teacherToStudent: RunTransducer;
}
export interface PairedPhoneRecording {
  written: string;
  teacher: string;
  student: string;
}

const sum = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0);
export function observationLogAdd(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const hi = Math.max(a, b);
  return hi + Math.log1p(Math.exp(Math.min(a, b) - hi));
}
export function validateRunTransducer(t: RunTransducer, maxFragmentLength = 2): void {
  if (!t.phones.length || new Set(t.phones).size !== t.phones.length || t.phones.some(p => [...p].length !== 1)
    || !t.fragments.includes("") || new Set(t.fragments).size !== t.fragments.length
    || t.fragments.some(f => [...f].length > maxFragmentLength || [...f].some(p => !t.phones.includes(p)))
    || t.rows.length !== t.phones.length || t.rows.some(r => r.length !== t.fragments.length
      || r.some(p => !Number.isFinite(p) || p < 0) || Math.abs(sum(r) - 1) > 1e-8)) {
    throw new Error("Invalid normalized run transducer");
  }
}

/** Sparse polynomial composition: coefficients sum all intermediate teacher strings and output splits. */
export function composeRunTransducers(model: TeacherObservationModel): RunTransducer {
  const a = model.referenceToTeacher, b = model.teacherToStudent;
  validateRunTransducer(a); validateRunTransducer(b);
  if (a.phones.join("") !== b.phones.join("")) throw new Error("Teacher inventory differs between stages");
  const phone = new Map(b.phones.map((p, i) => [p, i]));
  const expansions = a.fragments.map(fragment => {
    let terms = new Map<string, number>([["", 1]]);
    for (const p of fragment) {
      const next = new Map<string, number>(), row = b.rows[phone.get(p)!];
      for (const [prefix, probability] of terms) for (let j = 0; j < b.fragments.length; j++) {
        if (!row[j]) continue;
        const output = prefix + b.fragments[j];
        next.set(output, (next.get(output) ?? 0) + probability * row[j]);
      }
      terms = next;
    }
    return terms;
  });
  const fragments = [...new Set(expansions.flatMap(m => [...m.keys()]))].sort((x, y) => x.length - y.length || x.localeCompare(y));
  const fi = new Map(fragments.map((f, i) => [f, i]));
  const rows = a.rows.map(row => {
    const out = Array(fragments.length).fill(0) as number[];
    for (let k = 0; k < row.length; k++) if (row[k]) for (const [f, probability] of expansions[k]) out[fi.get(f)!] += row[k] * probability;
    return out;
  });
  // Composed fragments can have four phones, so use a separate validation contract.
  const composed = { phones: [...a.phones], fragments, rows };
  validateRunTransducer(composed, 4);
  return composed;
}

/** Lazy composition avoids enumerating every possible four-phone student string at inference. */
export function createTeacherObservationScorer(model: TeacherObservationModel) {
  const a = model.referenceToTeacher, b = model.teacherToStudent;
  validateRunTransducer(a); validateRunTransducer(b);
  if (a.phones.join("") !== b.phones.join("")) throw new Error("Teacher inventory differs between stages");
  const pi = new Map(a.phones.map((p, i) => [p, i]));
  const bi = new Map(b.fragments.map((f, i) => [f, i]));
  const cache = new Map<string, number[]>();
  type Prefix = { parent: Prefix | null; phone: number; values: number[] };
  let prefixCache = new Map<string, Prefix>();
  let observed = "";
  const second = (p: string, output: string) => b.rows[pi.get(p)!][bi.get(output) ?? -1] ?? 0;
  const coefficient = (output: string): number[] => {
    const cached = cache.get(output);
    if (cached) return cached;
    const latent = a.fragments.map(f => {
      if (!f) return Number(!output);
      if (f.length === 1) return second(f, output);
      let probability = 0;
      for (let j = 0; j <= output.length; j++) probability += second(f[0], output.slice(0, j)) * second(f[1], output.slice(j));
      return probability;
    });
    const rows = a.rows.map(r => r.reduce((v, p, j) => v + p * latent[j], 0));
    // Cache only fragments, never passages or recording identities.
    if (cache.size >= 8192) cache.clear();
    cache.set(output, rows);
    return rows;
  };
  return {
    probability(referencePhone: string, output: string) {
      const i = pi.get(referencePhone);
      if (i === undefined || [...output].some(p => !pi.has(p))) throw new Error("Phone outside observation inventory");
      return output.length <= 4 ? coefficient(output)[i] : 0;
    },
    /** Exact full-string likelihood; sums all run-to-token segmentations. */
    logP(written: string, student: string): number {
      if ([...written, ...student].some(p => !pi.has(p))) throw new Error("Phone outside observation inventory");
      // Prefix rows are immutable under observation extension. Rewinds/forks invalidate the memo rather than
      // letting a previous token branch influence the score. Cache eviction affects work only, never likelihood.
      if (!student.startsWith(observed) || prefixCache.size > 8192) prefixCache = new Map();
      observed = student;
      let root = prefixCache.get("");
      if (!root) { root = { parent: null, phone: -1, values: [0] }; prefixCache.set("", root); }
      while (root.values.length <= student.length) root.values.push(-Infinity);
      let previous = root, key = "";
      for (const p of written) {
        key += p;
        let current = prefixCache.get(key);
        if (!current) { current = { parent: previous, phone: pi.get(p)!, values: [] }; prefixCache.set(key, current); }
        for (let j = current.values.length; j <= student.length; j++) {
          let value = -Infinity;
          for (let length = 0; length <= Math.min(4, j); length++) {
            const probability = coefficient(student.slice(j - length, j))[current.phone];
            if (probability) value = observationLogAdd(value, previous.values[j - length] + Math.log(probability));
          }
          current.values.push(value);
        }
        previous = current;
      }
      return previous.values[student.length];
    },
  };
}

function initial(phones: string[], fragments: string[]): RunTransducer {
  return { phones, fragments, rows: phones.map(p => {
    const row = fragments.map(f => f === p ? 8 : f === "" ? 1 : 0.01);
    const z = sum(row);
    return row.map(x => x / z);
  }) };
}
function smoothed(t: RunTransducer, counts: number[][], mass: number): RunTransducer {
  // Pool by output fragment and retain an identity prior; no Quran-word or speaker parameters.
  const pooled = t.fragments.map((_, j) => counts.reduce((v, r) => v + r[j], 0) + 1);
  const pooledTotal = sum(pooled);
  return { ...t, rows: counts.map((r, i) => {
    const prior = pooled.map((v, j) => v + Number(t.fragments[j] === t.phones[i]) * pooledTotal);
    const z = sum(prior), total = sum(r) + mass;
    return r.map((v, j) => (v + mass * prior[j] / z) / total);
  }) };
}

/** Forward/backward with epsilon output and bounded fragments; each input run is consumed exactly once. */
function expectedSegments(written: string, output: string, probability: (p: string, f: string) => number,
  maxLength: number, visit: (p: string, f: string, weight: number) => void): number {
  const width = output.length + 1, at = (i: number, j: number) => i * width + j;
  const a = new Float64Array((written.length + 1) * width).fill(-Infinity), b = new Float64Array(a.length).fill(-Infinity);
  a[0] = 0; b[at(written.length, output.length)] = 0;
  for (let i = 0; i < written.length; i++) for (let j = 0; j <= output.length; j++) {
    for (let k = 0; k <= Math.min(maxLength, output.length - j); k++) {
      const v = probability(written[i], output.slice(j, j + k));
      if (v) a[at(i + 1, j + k)] = observationLogAdd(a[at(i + 1, j + k)], a[at(i, j)] + Math.log(v));
    }
  }
  const logP = a[at(written.length, output.length)];
  if (!Number.isFinite(logP)) throw new Error("Training string not supported by fragment graph");
  for (let i = written.length - 1; i >= 0; i--) for (let j = output.length; j >= 0; j--) {
    for (let k = 0; k <= Math.min(maxLength, output.length - j); k++) {
      const fragment = output.slice(j, j + k), v = probability(written[i], fragment);
      if (!v) continue;
      b[at(i, j)] = observationLogAdd(b[at(i, j)], Math.log(v) + b[at(i + 1, j + k)]);
      visit(written[i], fragment, Math.exp(a[at(i, j)] + Math.log(v) + b[at(i + 1, j + k)] - logP));
    }
  }
  return logP;
}

function fitStage(records: readonly { written: string; output: string }[], seed: RunTransducer, mass: number, iterations: number): RunTransducer {
  let model = seed;
  const pi = new Map(seed.phones.map((p, i) => [p, i])), fi = new Map(seed.fragments.map((f, i) => [f, i]));
  for (let iteration = 0; iteration < iterations; iteration++) {
    const counts = model.rows.map(r => r.map(() => 0));
    for (const record of records) expectedSegments(record.written, record.output,
      (p, f) => model.rows[pi.get(p)!][fi.get(f) ?? -1] ?? 0, 2,
      (p, f, weight) => { counts[pi.get(p)!][fi.get(f)!] += weight; });
    model = smoothed(model, counts, mass);
  }
  return model;
}

/** Both fits have exactly the same fragment support, two tables, pooling rule and EM iteration budget. */
export function fitTeacherObservationModels(records: readonly PairedPhoneRecording[], phones: string[], mass: number, iterations = 16) {
  if (!records.length || !(mass > 0) || !Number.isInteger(iterations) || iterations < 1) throw new Error("Invalid observation training configuration");
  if (records.some(r => !r.written || !r.teacher || !r.student || [...r.written, ...r.teacher, ...r.student].some(p => !phones.includes(p)))) throw new Error("Invalid paired training recording");
  const fragments = ["", ...phones, ...new Set(records.flatMap(r => [r.teacher, r.student].flatMap(s =>
    Array.from({ length: Math.max(0, s.length - 1) }, (_, i) => s.slice(i, i + 2)))))];
  const seed = initial(phones, fragments);
  const mediated: TeacherObservationModel = {
    referenceToTeacher: fitStage(records.map(r => ({ written: r.written, output: r.teacher })), seed, mass, iterations),
    teacherToStudent: fitStage(records.map(r => ({ written: r.teacher, output: r.student })), seed, mass, iterations),
  };
  // Direct control: the same factorized reference-to-student family, trained without observing its latent runs.
  let direct: TeacherObservationModel = { referenceToTeacher: seed, teacherToStudent: seed };
  const pi = new Map(phones.map((p, i) => [p, i])), fi = new Map(fragments.map((f, i) => [f, i]));
  for (let iteration = 0; iteration < iterations; iteration++) {
    const a = direct.referenceToTeacher, b = direct.teacherToStudent;
    const ac = a.rows.map(r => r.map(() => 0)), bc = b.rows.map(r => r.map(() => 0));
    const scorer = createTeacherObservationScorer(direct);
    for (const record of records) expectedSegments(record.written, record.student, scorer.probability, 4, (p, output, weight) => {
      const r = pi.get(p)!, total = scorer.probability(p, output);
      if (!weight || !total) return;
      const second = (phone: string, f: string) => b.rows[pi.get(phone)!][fi.get(f) ?? -1] ?? 0;
      for (let k = 0; k < fragments.length; k++) {
        const latent = fragments[k];
        if (!latent) { if (!output) ac[r][k] += weight * a.rows[r][k] / total; continue; }
        for (let split = 0; split <= (latent.length === 1 ? 0 : output.length); split++) {
          const x = latent.length === 1 ? output : output.slice(0, split), y = output.slice(split);
          const v = weight * a.rows[r][k] * second(latent[0], x) * (latent.length === 1 ? 1 : second(latent[1], y)) / total;
          if (!v) continue;
          ac[r][k] += v; bc[pi.get(latent[0])!][fi.get(x)!] += v;
          if (latent.length === 2) bc[pi.get(latent[1])!][fi.get(y)!] += v;
        }
      }
    });
    direct = { referenceToTeacher: smoothed(a, ac, mass), teacherToStudent: smoothed(b, bc, mass) };
  }
  return { mediated, direct, parameters: 2 * phones.length * (fragments.length - 1) };
}

/** Same conversion as the tracker's posterior ranking; no fitted inference threshold. */
export const observationCost = (logP: number) => -logP * UNIT / 3;

export interface ObservationPath {
  /** Reference phones traversed by the retained path; jump intervals are absent. */
  written: string;
  /** Existing accumulated cost, including movement and duration charges, in UNITs. */
  cost: number;
}
export interface TeacherObservationComparison {
  readonly n: number;
  readonly heard: string;
  readonly continuation: Readonly<ObservationPath & { teacherLogP: number; directLogP: number; correction: number; total: number }>;
  readonly restart: Readonly<ObservationPath & { teacherLogP: number; directLogP: number; correction: number; total: number }>;
  readonly winner: "continuation" | "restart" | null;
}

/** Training-frozen residual compared over one shared, complete observation interval. */
export function createTeacherObservationComparison(model: { mediated: TeacherObservationModel; direct: TeacherObservationModel }) {
  const mediated = createTeacherObservationScorer(model.mediated), direct = createTeacherObservationScorer(model.direct);
  return (continuation: ObservationPath, restart: ObservationPath, heard: string): TeacherObservationComparison | null => {
    const score = (path: ObservationPath) => {
      const teacherLogP = mediated.logP(path.written, heard), directLogP = direct.logP(path.written, heard);
      if (!Number.isFinite(teacherLogP) || !Number.isFinite(directLogP)) return null;
      const correction = observationCost(teacherLogP - directLogP);
      return Object.freeze({ ...path, teacherLogP, directLogP, correction, total: path.cost + correction });
    };
    const c = score(continuation), r = score(restart);
    if (!c || !r) return null;
    return Object.freeze({ n: heard.length, heard, continuation: c, restart: r,
      winner: c.total < r.total ? "continuation" as const : r.total < c.total ? "restart" as const : null });
  };
}
