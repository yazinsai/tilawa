import type { DecodedPhoneme, PauseMaskEvidence, PhonemeEvidence } from "./evidence.js";

/** Read alternate rows only at their declared frames. Primary evidence is never
 * mutated. Tokens wait for their complete run to arrive on the primary clock. */
export function pauseEvidenceReader(tokens: ReadonlySet<string>) {
  let spans: { evidence: PauseMaskEvidence; emitted: Set<number> }[] = [];
  const seen = new Set<string>();
  return (primary: PhonemeEvidence): PhonemeEvidence => {
    const certificates: import("./evidence.js").SeamInsertionEvidence[] = [];
    const rows = primary.posteriors;
    if (!rows) return primary;
    for (const p of primary.phonemes) for (const s of p.pauseMaskEvidence ?? []) {
      const key = `${s.fromFrame}:${s.toFrame}`;
      if (seen.has(key)) continue;
      const r = s.posteriors;
      if (s.inventory !== primary.inventory || s.frameMs !== primary.frameMs ||
        !Number.isInteger(s.fromFrame) || s.fromFrame < 0 || !Number.isInteger(s.toFrame) || s.toFrame <= s.fromFrame ||
        r.startFrame !== s.fromFrame || r.nFrames !== s.toFrame-s.fromFrame || r.k !== rows.k ||
        r.ids.length !== r.nFrames*r.k || r.probs.length !== r.nFrames*r.k ||
        s.phonemes.some(p => !tokens.has(p.p) || !Number.isInteger(p.start) || !Number.isInteger(p.end) ||
          p.start < s.fromFrame || p.end >= s.toFrame || p.end < p.start || !Number.isFinite(p.conf) || p.conf < 0 || p.conf > 1)) continue;
      let valid = true;
      for (let i = 0; i < r.ids.length; i++) if (!Number.isInteger(r.ids[i]) || r.ids[i] < 0 || r.ids[i] > tokens.size ||
        !Number.isFinite(r.probs[i]) || r.probs[i] < 0 || r.probs[i] > 1) { valid = false; break; }
      if (!valid || spans.some(x => x.evidence.fromFrame < s.toFrame && x.evidence.toFrame > s.fromFrame)) continue;
      seen.add(key); spans.push({ evidence: s, emitted: new Set() });
      certificates.push(...(s.seamInsertions ?? []));
    }
    if (certificates.length) primary = { ...primary,phonemes: primary.phonemes.map((p,i) => i ? p : {
      ...p,seamInsertions:[...(p.seamInsertions ?? []),...certificates],
    }) };
    if (!spans.length) return primary;
    spans.sort((a,b) => a.evidence.fromFrame-b.evidence.fromFrame);
    const until = rows.startFrame+rows.nFrames;
    const active = spans.filter(x => x.evidence.fromFrame < until && x.evidence.toFrame > rows.startFrame);
    if (!active.length) return primary;
    const ids = Array.from(rows.ids), probs = Array.from(rows.probs);
    for (const { evidence: s } of active) {
      const from = Math.max(rows.startFrame,s.fromFrame), end = Math.min(until,s.toFrame);
      for (let f = from; f < end; f++) for (let j = 0; j < rows.k; j++) {
        const dst = (f-rows.startFrame)*rows.k+j, src = (f-s.fromFrame)*rows.k+j;
        ids[dst] = s.posteriors.ids[src]; probs[dst] = s.posteriors.probs[src];
      }
    }
    const phonemes: DecodedPhoneme[] = [];
    for (const p of primary.phonemes) {
      let parts = [{ ...p }];
      for (const { evidence: s } of active) parts = parts.flatMap(q => {
        if (q.end < s.fromFrame || q.start >= s.toFrame) return [q];
        return [
          ...(q.start < s.fromFrame ? [{ ...q,end: s.fromFrame-1 }] : []),
          ...(q.end >= s.toFrame ? [{ ...q,start: s.toFrame }] : []),
        ];
      });
      phonemes.push(...parts);
    }
    for (const x of active) x.evidence.phonemes.forEach((p,i) => {
      if (!x.emitted.has(i) && p.end < until) { phonemes.push({ ...p }); x.emitted.add(i); }
    });
    spans = spans.filter(x => x.evidence.toFrame > until);
    return { ...primary,phonemes: phonemes.sort((a,b) => a.start-b.start || a.end-b.end),
      posteriors: { ...rows,ids,probs } };
  };
}
