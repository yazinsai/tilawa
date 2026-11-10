import { PosteriorFrames } from "./ctc.js";

/** Work bound, not an acoustic threshold: longer intervals retain decoded tracking. */
export const MAX_LATTICE_FRAMES = 512;

export interface AcousticPrefix {
  phones: string;
  /** Total probability of all CTC frame paths for this string, in log units. */
  logProbability: number;
  /** A supported frame path's collapsed runs; equivalent boundary paths share the probability above. */
  runs: { p: string; start: number; end: number }[];
}

const add = (a: number, b: number) => a === -Infinity ? b : b === -Infinity ? a
  : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

interface Path { score: number; runs: AcousticPrefix["runs"] }
interface Prefix { phones: string; blank: number; nonblank: number; bp: Path; np: Path }
const emptyPath = (): Path => ({ score: -Infinity, runs: [] });

/**
 * Bounded CTC prefix search over one common acoustic interval. Blank and nonblank endings are separate, so a
 * repeated label extends its run unless a blank separates two articulations. Only listed labels create prefixes.
 * Beam pruning bounds string discovery; the final chain forward sum restores every boundary path of each retained
 * string, including paths pruned during discovery. Each string is scored once, never once per boundary history.
 */
export function acousticPrefixes(frames: PosteriorFrames, tokens: readonly string[], from: number, to: number, width = 64): AcousticPrefix[] {
  if (from < frames.startFrame || to > frames.endFrame || from > to || to - from > MAX_LATTICE_FRAMES || width < 1) return [];
  let beam: Prefix[] = [{ phones: "", blank: 0, nonblank: -Infinity, bp: { score: 0, runs: [] }, np: emptyPath() }];
  for (let f = from; f < to; f++) {
    const row = frames.row(f);
    const next = new Map<string, Prefix>();
    const get = (phones: string) => {
      let p = next.get(phones);
      if (!p) next.set(phones, p = { phones, blank: -Infinity, nonblank: -Infinity, bp: emptyPath(), np: emptyPath() });
      return p;
    };
    const labels = tokens.map((p, i) => ({ p, id: i + 1 })).filter(({ id }) => frames.lists(f, id));
    for (const old of beam) {
      const same = get(old.phones);
      const total = add(old.blank, old.nonblank);
      same.blank = add(same.blank, total + row[0]);
      const path = old.bp.score >= old.np.score ? old.bp : old.np;
      if (path.score + row[0] > same.bp.score) same.bp = { score: path.score + row[0], runs: path.runs };
      for (const { p, id } of labels) {
        const repeated = p === old.phones.at(-1);
        if (repeated) {
          same.nonblank = add(same.nonblank, old.nonblank + row[id]);
          if (old.np.score + row[id] > same.np.score) {
            const runs = old.np.runs.map((r) => ({ ...r }));
            runs[runs.length - 1].end = f;
            same.np = { score: old.np.score + row[id], runs };
          }
        }
        const ext = get(old.phones + p);
        ext.nonblank = add(ext.nonblank, (repeated ? old.blank : total) + row[id]);
        const ep = repeated ? old.bp : path;
        if (ep.score + row[id] > ext.np.score) ext.np = { score: ep.score + row[id], runs: [...ep.runs, { p, start: f, end: f }] };
      }
    }
    beam = [...next.values()].sort((a, b) => add(b.blank, b.nonblank) - add(a.blank, a.nonblank)).slice(0, width);
  }
  const ids = new Map(tokens.map((p, i) => [p, i + 1]));
  return beam.map((p) => ({ phones: p.phones, logProbability: -frames.nll([...p.phones].map((ph) => ids.get(ph)!), from, to),
    runs: (p.bp.score >= p.np.score ? p.bp : p.np).runs })).sort((a, b) => b.logProbability - a.logProbability);
}
