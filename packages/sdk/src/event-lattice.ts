import type { DecodedPhoneme } from "@tilawa/core";

export interface LatticeLimits { prefixes: number; labels: number; frames: number }
export interface Transcription { phonemes: DecodedPhoneme[]; logMass: number }
interface State { labels: number[]; blank: boolean; mass: number; best: number; path: number[] }
const add = (a: number, b: number) => a === -Infinity ? b : b === -Infinity ? a :
  Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

/** Reference-free CTC prefix lattice. Blank/nonblank endpoints sum all retained paths of each collapsed string.
 * The maximum-probability path supplies timing only; its score never replaces the summed string mass. */
export function transcriptionLattice(rows: readonly Float64Array[], tokens: readonly string[], from: number,
  limits: LatticeLimits): Transcription[] {
  if (!rows.length || rows.length > limits.frames) return [];
  if (limits.prefixes < 1 || limits.labels < 1) throw new RangeError("Positive lattice limits required");
  let states = new Map<string, State>([["/b", { labels: [], blank: true, mass: 0, best: 0, path: [] }]]);
  const normalized: Float64Array[] = [];
  for (const row of rows) {
    if (row.length !== tokens.length + 1 || row.some(p => !Number.isFinite(p) && p !== -Infinity)) return [];
    const normalizer = row.reduce(add, -Infinity);
    if (!Number.isFinite(normalizer)) return [];
    normalized.push(row.map(p => p - normalizer));
    const choices = Array.from(row, (p, id) => ({ id, p: p - normalizer }))
      .sort((a, b) => b.p - a.p || a.id - b.id).slice(0, limits.labels);
    if (!choices.some(c => c.id === 0)) choices.push({ id: 0, p: row[0] - normalizer });
    const next = new Map<string, State>();
    for (const s of states.values()) for (const { id, p } of choices) {
      if (p === -Infinity) continue;
      const labels = id === 0 || (!s.blank && id === s.labels.at(-1)) ? s.labels : [...s.labels, id];
      const key = `${labels.join(",")}/${id === 0 ? "b" : "n"}`;
      const old = next.get(key), mass = s.mass + p, best = s.best + p;
      if (!old) next.set(key, { labels, blank: id === 0, mass, best, path: [...s.path, id] });
      else {
        old.mass = add(old.mass, mass);
        if (best > old.best) { old.best = best; old.path = [...s.path, id]; }
      }
    }
    const prefixes = new Map<string, number>();
    for (const s of next.values()) {
      const key = s.labels.join(","); prefixes.set(key, add(prefixes.get(key) ?? -Infinity, s.mass));
    }
    const retained = new Set([...prefixes].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, limits.prefixes).map(([key]) => key));
    states = new Map([...next].filter(([, s]) => retained.has(s.labels.join(","))));
  }
  const merged = new Map<string, State>();
  for (const s of states.values()) {
    const key = s.labels.join(","), old = merged.get(key);
    if (!old) merged.set(key, { ...s });
    else { old.mass = add(old.mass, s.mass); if (s.best > old.best) { old.best = s.best; old.path = s.path; } }
  }
  return [...merged.values()].sort((a, b) => b.mass - a.mass).map(s => {
    const phonemes: DecodedPhoneme[] = [];
    let prev = 0;
    for (let f = 0; f < s.path.length; f++) {
      const id = s.path[f], conf = Math.exp(normalized[f][id]);
      if (id) {
        if (id === prev) { const last = phonemes.at(-1)!; last.end = from + f; last.conf = Math.max(last.conf, conf); }
        else phonemes.push({ p: tokens[id - 1], start: from + f, end: from + f, conf });
      }
      prev = id;
    }
    // Beam pruning selects strings, not probabilities: recover every CTC timing path of each retained string.
    return { phonemes, logMass: sequenceMass(normalized, s.labels) };
  });
}

function sequenceMass(rows: readonly Float64Array[], labels: readonly number[]): number {
  const size = 2 * labels.length + 1;
  const id = (s: number) => s % 2 ? labels[(s - 1) >> 1] : 0;
  let prev = new Float64Array(size).fill(-Infinity);
  prev[0] = 0;
  for (const row of rows) {
    const next = new Float64Array(size).fill(-Infinity);
    for (let s = 0; s < size; s++) {
      let mass = prev[s];
      if (s) mass = add(mass, prev[s - 1]);
      if (s > 1 && s % 2 && id(s) !== id(s - 2)) mass = add(mass, prev[s - 2]);
      next[s] = mass + row[id(s)];
    }
    prev = next;
  }
  return labels.length ? add(prev[size - 1], prev[size - 2]) : prev[0];
}

export interface EventMass { quiet: number; warning: number; retained: number; transcripts: number; classes: Record<string, number> }
/** The posterior is over warning events, never over a selected spelling or rival identity. */
export function warningEventMass(lattice: Transcription[], judge: (phones: DecodedPhoneme[]) => readonly string[]): EventMass {
  if (!lattice.length) return { quiet: 0, warning: 0, retained: 0, transcripts: 0, classes: {} };
  const classes: Record<string, number> = {};
  let quiet = -Infinity, warning = -Infinity;
  for (const t of lattice) {
    const events = judge(t.phonemes);
    if (events.length) {
      warning = add(warning, t.logMass);
      const key = [...new Set(events)].sort().join("+"); classes[key] = add(classes[key] ?? -Infinity, t.logMass);
    } else quiet = add(quiet, t.logMass);
  }
  const total = add(quiet, warning);
  return { quiet: Math.exp(quiet - total), warning: Math.exp(warning - total), retained: Math.exp(total),
    transcripts: lattice.length, classes: Object.fromEntries(Object.entries(classes).map(([k, p]) => [k, Math.exp(p - total)])) };
}
