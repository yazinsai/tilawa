import type { ReferenceIndex } from "./reference.js";

export type MorphemeRole = "prefix" | "stem" | "suffix";
export interface MorphemeUnit { role: MorphemeRole; phones: string }
export interface ChannelReference { phones: string; roles: readonly MorphemeRole[] }

const letters = (s: string) => s.replace(/[\u064b-\u065f\u0670\u06d6-\u06ed]/g, "").replace(/ٱ/g, "ا");
const vowel = (p: string) => "َُِاۦۥٲ۪".includes(p);
const consonant = (p: string) => !vowel(p);

/**
 * Conservative attached-clitic analysis, not a new pronunciation. Every unit is a slice of the supplied surface
 * phones. Unresolved orthographic/phonetic boundaries stay in the stem. The three-letter residue prevents
 * splitting intrinsic onsets such as ورد and فتح; this is deliberately less complete than a morphological lexicon.
 */
export function morphemeUnits(text: string, surface: string, unattached?: ReadonlySet<string>): MorphemeUnit[] {
  let spelling = letters(text);
  let from = 0;
  const prefix: MorphemeUnit[] = [];
  const take = (role: MorphemeRole, to: number) => {
    if (to > from) prefix.push({ role, phones: surface.slice(from, to) });
    from = to;
  };
  // A conjunction can precede another clitic. Require its explicitly vocalized spelling.
  const remainder = spelling.slice(1);
  const supportedConjunction = !unattached || unattached.has(remainder) || /^(?:[بكل]?ال|لل)/.test(remainder);
  if (/^[وف]َ/.test(text) && spelling.length >= 4 && surface[0] === spelling[0] && supportedConjunction) {
    const n = surface[1] === "َ" ? 2 : 1;
    take("prefix", n);
    spelling = spelling.slice(1);
  }
  // Prepositions are attached only when a definite article makes the analysis unambiguous.
  if (/^[بكل]ال/.test(spelling) || /^لل/.test(spelling)) {
    const p = spelling[0];
    if (surface[from] === p) {
      const to = from + 1 + ("َُِ".includes(surface[from + 1] ?? " ") ? 1 : 0);
      take("prefix", to);
      spelling = spelling.slice(1);
    }
  }
  const article = spelling.startsWith("ال") ? 2 : spelling.startsWith("ل") && /^لل/.test(letters(text).replace(/^[وف]/, "")) ? 1 : 0;
  if (article && spelling.length >= article + 3) {
    let to = from;
    // Hamzat-wasl is already elided in connected reference forms; retain it when it is present.
    if (surface[to] === "ء" && "َُِ".includes(surface[to + 1] ?? " ")) to += 2;
    const stemLetter = spelling[article];
    if (surface[to] === "ل") to++;
    else if (surface[to] === stemLetter) {
      let end = to;
      while (surface[end] === stemLetter) end++;
      // Assimilated article contributes the extra onset copies; one consonant remains in the stem.
      if (end - to > 1) to = end - 1;
    }
    if (to > from) { take("prefix", to); spelling = spelling.slice(article); }
  }
  // Attached personal pronouns, with a substantial stem. Locate their surviving consonants in the surface tail.
  const suffixes = ["هما", "كما", "هم", "هن", "كم", "كن", "نا", "ها", "ه", "ك"];
  for (const s of suffixes) {
    if (!spelling.endsWith(s) || spelling.length - s.length < 3) continue;
    const wanted = [...s].filter(consonant);
    let k = surface.length - 1;
    let begin = -1;
    let good = true;
    for (let j = wanted.length - 1; j >= 0; j--) {
      while (k >= from && vowel(surface[k])) k--;
      if (surface[k] !== wanted[j]) { good = false; break; }
      begin = k--;
    }
    if (good && begin > from) {
      prefix.push({ role: "stem", phones: surface.slice(from, begin) });
      prefix.push({ role: "suffix", phones: surface.slice(begin) });
      return prefix;
    }
  }
  if (from < surface.length) prefix.push({ role: "stem", phones: surface.slice(from) });
  return prefix;
}

const lexicons = new WeakMap<ReferenceIndex, ReadonlySet<string>>();
const fullReferences = new WeakMap<ReferenceIndex, ChannelReference>();
export function channelReference(index: ReferenceIndex, from: number, to: number): ChannelReference {
  const whole = from === 0 && to === index.phonemes.length;
  const cached = whole ? fullReferences.get(index) : undefined;
  if (cached) return cached;
  let unattached = lexicons.get(index);
  if (!unattached) {
    unattached = new Set((index.wordText ?? []).map(letters));
    lexicons.set(index, unattached);
  }
  const roles: MorphemeRole[] = [];
  for (let i = index.wordAt(from); i < index.words.length && index.words[i].start < to; i++) {
    const word = index.words[i];
    const units = morphemeUnits(index.wordText?.[i] ?? "", index.phonemes.slice(word.start, word.end), unattached);
    const wordRoles = units.flatMap(u => Array<MorphemeRole>(u.phones.length).fill(u.role));
    roles.push(...wordRoles.slice(Math.max(0, from - word.start), Math.min(wordRoles.length, to - word.start)));
  }
  const result = { phones: index.phonemes.slice(from, to), roles: Object.freeze(roles) };
  if (whole) fullReferences.set(index, result);
  return result;
}

/** Shared features, never a word identity or passage location. */
export function phoneClass(p: string | undefined): string {
  if (!p) return "edge";
  if ("َُِ۪".includes(p)) return "vowel";
  if ("اۦۥٲ".includes(p)) return "long";
  if ("نمں۾".includes(p)) return "nasal";
  if ("لروي".includes(p)) return "sonorant";
  if ("بتدطضكقء".includes(p)) return "stop";
  return "fricative";
}

export type ChannelKind = "phonetic" | "morphology";
/** Both models receive identical phones. Only the morphology model receives clitic roles. */
export function channelContext(ref: ChannelReference, at: number, kind: ChannelKind): string {
  const p = ref.phones;
  const context = `${phoneClass(p[at - 1])}/${phoneClass(p[at])}/${phoneClass(p[at + 1])}/${p[at] === p[at - 1] ? "held" : "new"}`;
  return kind === "phonetic" ? context : `${context}/${ref.roles[at - 1] ?? "edge"}>${ref.roles[at] ?? "stem"}`;
}

/** The preceding reference event: correct identity, substitution, or deletion. */
export type EditState = 0 | 1 | 2;
export interface MorphemeChannel {
  kind: ChannelKind;
  phones: string;
  /** Geometric insertion probability; each insertion consumes one new observation. */
  insertion: number;
  inserted: number[];
  /** Rows exclude the canonical identity and sum to one over all other phones. */
  substituted: number[][];
  pooled: number[][];
  /** Context rows contain P(next edit | previous edit, reference context). */
  contexts: Record<string, number[][]>;
  strength: number;
}

export interface TrainingTake { reference: ChannelReference; heard: string; speaker: string }
export interface ExpectedCounts {
  edits: number[][];
  contexts: Record<string, number[][]>;
  insertions: number[];
  substitutions: number[][];
  advances: number;
}
const matrix = (rows: number, cols: number) => Array.from({ length: rows }, () => Array<number>(cols).fill(0));
const emptyCounts = (n: number): ExpectedCounts => ({ edits: matrix(3, 3), contexts: {}, insertions: Array(n).fill(0), substitutions: matrix(n, n), advances: 0 });
const addLog = (a: number, b: number) => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log1p(Math.exp(Math.min(a, b) - m));
};

/**
 * A proper conditional observation transducer. Each reference advance chooses a normalized C/S/D event;
 * insertions form a terminating geometric loop. The edit state carries across unit boundaries, so errors on
 * either side are jointly modelled. Forward/backward sums token placement rather than trusting tracker visits.
 * Reference length repetitions remain in the string. No unit has an epsilon bypass or a free deletion.
 */
export function channelLogLikelihood(ref: ChannelReference, heard: string, model: MorphemeChannel, counts?: ExpectedCounts): number {
  if (ref.roles.length !== ref.phones.length) throw new Error("Surface phone/role length mismatch");
  const n = ref.phones.length, m = heard.length, width = (m + 1) * 3;
  const at = (i: number, j: number, state: number) => i * width + j * 3 + state;
  const f = new Float64Array((n + 1) * width).fill(-Infinity);
  const b = counts ? new Float64Array(f.length).fill(-Infinity) : null;
  const pi = new Map([...model.phones].map((p, i) => [p, i]));
  const wi = [...ref.phones].map(p => pi.get(p) ?? -1);
  const hi = [...heard].map(p => pi.get(p) ?? -1);
  if (wi.includes(-1) || hi.includes(-1)) throw new Error("Unknown observation-channel phone");
  const keys = Array.from({ length: n }, (_, i) => channelContext(ref, i, model.kind));
  const rows = keys.map(k => model.contexts[k] ?? model.pooled);
  const logInsert = Math.log(model.insertion), logAdvance = Math.log1p(-model.insertion);
  const edges = (i: number, j: number, state: number, visit: (to: number, logp: number, op: number) => void) => {
    if (j < m) visit(at(i, j + 1, state), logInsert + Math.log(model.inserted[hi[j]]), 3);
    if (i < n) {
      visit(at(i + 1, j, 2), logAdvance + Math.log(rows[i][state][2]), 2);
      if (j < m) {
        const op = wi[i] === hi[j] ? 0 : 1;
        visit(at(i + 1, j + 1, op), logAdvance + Math.log(rows[i][state][op])
          + (op === 0 ? 0 : Math.log(model.substituted[wi[i]][hi[j]])), op);
      }
    }
  };
  f[0] = 0;
  for (let i = 0; i <= n; i++) for (let j = 0; j <= m; j++) for (let state = 0; state < 3; state++) {
    const value = f[at(i, j, state)];
    if (value !== -Infinity) edges(i, j, state, (to, logp) => { f[to] = addLog(f[to], value + logp); });
  }
  let logp = -Infinity;
  for (let state = 0; state < 3; state++) logp = addLog(logp, f[at(n, m, state)] + logAdvance);
  if (!counts || !b) return logp;
  for (let state = 0; state < 3; state++) b[at(n, m, state)] = logAdvance;
  for (let i = n; i >= 0; i--) for (let j = m; j >= 0; j--) for (let state = 0; state < 3; state++) {
    const source = at(i, j, state);
    edges(i, j, state, (to, lp, op) => {
      b[source] = addLog(b[source], lp + b[to]);
      const weight = Math.exp(f[source] + lp + b[to] - logp);
      if (op === 3) counts.insertions[hi[j]] += weight;
      else {
        counts.edits[state][op] += weight;
        const table = counts.contexts[keys[i]] ??= matrix(3, 3);
        table[state][op] += weight;
        counts.advances += weight;
        if (op === 1) counts.substitutions[wi[i]][hi[j]] += weight;
      }
    });
  }
  return logp;
}

const normalize = (row: number[]) => { const z = row.reduce((a, b) => a + b, 0); return row.map(x => x / z); };

/** Fixed structural prior. Its total mass is selected only on grouped training-speaker folds. */
export function initialChannel(phones: string, kind: ChannelKind, strength: number): MorphemeChannel {
  const n = phones.length;
  return { phones, kind, strength, insertion: 0.02, inserted: Array(n).fill(1 / n),
    substituted: [...phones].map((p, i) => normalize([...phones].map((q, j) => i === j ? 0 : phoneClass(p) === phoneClass(q) ? 2 : 1))),
    pooled: [[0.9, 0.05, 0.05], [0.7, 0.15, 0.15], [0.7, 0.15, 0.15]], contexts: {} };
}

export function fitMorphemeChannel(takes: readonly TrainingTake[], phones: string, kind: ChannelKind, strength: number, iterations = 6): MorphemeChannel {
  const prior = initialChannel(phones, kind, strength);
  let model = prior;
  for (let iteration = 0; iteration < iterations; iteration++) {
    const counts = emptyCounts(phones.length);
    for (const take of takes) channelLogLikelihood(take.reference, take.heard, model, counts);
    const pooled = counts.edits.map((row, i) => normalize(row.map((x, j) => x + strength * prior.pooled[i][j])));
    const contexts = Object.fromEntries(Object.entries(counts.contexts).map(([key, table]) => [key,
      table.map((row, i) => normalize(row.map((x, j) => x + strength * pooled[i][j])))]));
    const inserted = normalize(counts.insertions.map((x, i) => x + strength * prior.inserted[i]));
    const substituted = counts.substitutions.map((row, i) => normalize(row.map((x, j) => x + strength * prior.substituted[i][j])));
    const insertedCount = counts.insertions.reduce((a, b) => a + b, 0);
    model = { ...prior, pooled, contexts, inserted, substituted,
      insertion: (insertedCount + strength * prior.insertion) / (insertedCount + counts.advances + takes.length + strength) };
  }
  return model;
}

/** Same speaker partitions, fitting recipe and observations for the two channels; no practice data is accepted. */
export function selectMorphemeChannels(takes: readonly TrainingTake[], phones: string) {
  const speakers = [...new Set(takes.map(t => t.speaker))].sort();
  if (speakers.length < 3) throw new Error("Need at least three training speakers for grouped selection");
  const folds = new Map(speakers.map((s, i) => [s, i % 3]));
  const selection: { kind: ChannelKind; strength: number; heldoutLogP: number }[] = [];
  const models = {} as Record<ChannelKind, MorphemeChannel>;
  for (const kind of ["phonetic", "morphology"] as const) {
    let best = { strength: 0, heldoutLogP: -Infinity };
    for (const strength of [2, 16, 128]) {
      let heldoutLogP = 0;
      for (let fold = 0; fold < 3; fold++) {
        const train = takes.filter(t => folds.get(t.speaker) !== fold);
        const valid = takes.filter(t => folds.get(t.speaker) === fold);
        const model = fitMorphemeChannel(train, phones, kind, strength);
        for (const t of valid) heldoutLogP += channelLogLikelihood(t.reference, t.heard, model);
      }
      selection.push({ kind, strength, heldoutLogP });
      if (heldoutLogP > best.heldoutLogP) best = { strength, heldoutLogP };
    }
    models[kind] = fitMorphemeChannel(takes, phones, kind, best.strength);
  }
  return { models, selection, trainingTakes: takes.length, trainingSpeakers: speakers.length };
}
