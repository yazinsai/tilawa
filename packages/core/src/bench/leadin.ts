/**
 * Help clips in context: 1–2 correctly recited ayahs stitched in front (lead-in), so a planted error falls
 * mid-recitation, and 1–2 after (lead-out), so later correct words can confirm it inside the session.
 */
import type { ReferenceIndex } from "../reference.js";
import type { AcceptanceClip, ClipScore } from "./acceptance.js";
import { stitchSession } from "./jumps.js";
import type { ClipEvidence, Segment, SessionPart } from "./jumps.js";

export interface LeadIn {
  /** Source clip ID. */
  source: string;
  /** The ayahs it recites, in order: ending just before the clip's first ayah, or (lead-out) starting just after its last. */
  ayahs: string[];
  evidence: ClipEvidence;
}

/**
 * The clip with the lead-in's evidence in front of it and the lead-out's after it, each `gapMs` of silence apart. Its
 * passage runs from the first recited ayah to the last, so a declared session declares what the reciter recited; the
 * labels are the clip's own.
 */
export function withContext(
  index: ReferenceIndex,
  clip: AcceptanceClip,
  o: { lead?: LeadIn | null; follow?: LeadIn | null },
  gapMs: number,
): { clip: AcceptanceClip; leadAyahs: string[]; followAyahs: string[]; segments: Segment[] } {
  const part = (l: LeadIn, gapBeforeMs: number): SessionPart => ({ source: l.source, ayahs: l.ayahs, gapBeforeMs, gapFill: "silence", evidence: l.evidence });
  const parts: SessionPart[] = [
    ...(o.lead ? [part(o.lead, 0)] : []),
    { source: clip.id, ayahs: clip.ayahs, gapBeforeMs: o.lead ? gapMs : 0, gapFill: "silence", evidence: clip.evidence },
    ...(o.follow ? [part(o.follow, gapMs)] : []),
  ];
  const leadAyahs = o.lead?.ayahs ?? [];
  const followAyahs = o.follow?.ayahs ?? [];
  const s = stitchSession(index, parts);
  return { clip: { ...clip, ayahs: [...leadAyahs, ...clip.ayahs, ...followAyahs], evidence: s.evidence as AcceptanceClip["evidence"] }, leadAyahs, followAyahs, segments: s.segments };
}

/** The score restricted to the original clip's span; false positives on the lead-in's and lead-out's ayahs are counted apart. */
export function inClip(score: ClipScore, leadAyahs: string[], followAyahs: string[] = []): { score: ClipScore; leadInWarnings: number; leadOutWarnings: number } {
  const on = (ayahs: string[]) => {
    const set = new Set(ayahs);
    return (e: ClipScore["falsePositives"][number]) => set.has(`${e.location.surah}:${e.location.ayah}`);
  };
  const onLead = on(leadAyahs), onFollow = on(followAyahs);
  const keep = (es: ClipScore["falsePositives"]) => es.filter((e) => !onLead(e) && !onFollow(e));
  return {
    score: { ...score, final: keep(score.final), confirmed: keep(score.confirmed), falsePositives: keep(score.falsePositives) },
    leadInWarnings: score.falsePositives.filter(onLead).length,
    leadOutWarnings: score.falsePositives.filter(onFollow).length,
  };
}
