import type { AcousticPrefix } from "./ctc-prefix.js";
import { UNIT } from "./fuzzy.js";
import type { LocalCandidate, LocalSearch } from "./local.js";

export interface AcousticCandidate extends LocalCandidate {
  /** Marginal cost in the local search's existing edit units. */
  cost: number;
  /** Most likely string contributing to this frontier, for provenance. */
  phones: string;
}

/**
 * Compose discovered acoustic strings with the unchanged local edit/movement recurrence from a common checkpoint.
 * There is one minimum reference-edit explanation per string and destination. Sum across strings, whose exact CTC
 * boundary probabilities have already been summed by the acoustic lattice; never count alternative edits as extra
 * acoustic paths. This read-only comparison leaves the live search and its commitment evidence intact.
 */
export function localAcousticCandidates(search: LocalSearch, checkpoint: unknown, prefixes: readonly AcousticPrefix[], natsPerEdit: number): AcousticCandidate[] {
  const live = search.save();
  const scores = new Map<number, AcousticCandidate & { logMass: number; best: number }>();
  const seen = new Set<string>();
  try {
    for (const prefix of prefixes) {
      if (!Number.isFinite(prefix.logProbability)) continue;
      if (seen.has(prefix.phones)) continue;
      seen.add(prefix.phones);
      search.restore(checkpoint);
      for (const p of prefix.phones) search.step(p);
      for (const c of search.candidates()) {
        if (!Number.isFinite(c.cost)) continue;
        const term = prefix.logProbability - c.cost / UNIT * natsPerEdit;
        let s = scores.get(c.offset);
        if (!s) scores.set(c.offset, s = { ...c, phones: prefix.phones, logMass: -Infinity, best: -Infinity });
        s.logMass = s.logMass === -Infinity ? term : Math.max(s.logMass, term) + Math.log1p(Math.exp(-Math.abs(s.logMass - term)));
        if (term > s.best) { s.best = term; s.origin = c.origin; s.edits = c.edits; s.recent = c.recent; s.phones = prefix.phones; }
      }
    }
  } finally {
    search.restore(live);
  }
  return [...scores.values()].map(({ logMass, best: _best, ...c }) => ({ ...c, cost: -logMass / natsPerEdit * UNIT }));
}
