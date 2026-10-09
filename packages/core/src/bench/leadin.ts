/** Help clips with a lead-in: 1–2 correctly recited ayahs stitched in front, so a planted error falls mid-recitation. */
import type { ReferenceIndex } from "../reference.js";
import type { AcceptanceClip, ClipScore } from "./acceptance.js";
import { stitchSession } from "./jumps.js";
import type { ClipEvidence } from "./jumps.js";

export interface LeadIn {
  /** Source clip ID. */
  source: string;
  /** The ayahs the lead-in recites, in order, ending just before the clip's first ayah. */
  ayahs: string[];
  evidence: ClipEvidence;
}

/**
 * The clip with the lead-in's evidence in front of it after `gapMs` of silence. Its passage runs from the lead-in's first
 * ayah, so a declared session declares from where the reciter started; the labels are the clip's own.
 */
export function withLeadIn(index: ReferenceIndex, clip: AcceptanceClip, lead: LeadIn, gapMs: number): { clip: AcceptanceClip; leadAyahs: string[] } {
  const s = stitchSession(index, [
    { source: lead.source, ayahs: lead.ayahs, gapBeforeMs: 0, evidence: lead.evidence },
    { source: clip.id, ayahs: clip.ayahs, gapBeforeMs: gapMs, gapFill: "silence", evidence: clip.evidence },
  ]);
  return { clip: { ...clip, ayahs: [...lead.ayahs, ...clip.ayahs], evidence: s.evidence as AcceptanceClip["evidence"] }, leadAyahs: lead.ayahs };
}

/** The score restricted to the original clip's span; false positives on the lead-in's ayahs are counted apart. */
export function inClip(score: ClipScore, leadAyahs: string[]): { score: ClipScore; leadInWarnings: number } {
  const lead = new Set(leadAyahs);
  const onLead = (e: ClipScore["falsePositives"][number]) => lead.has(`${e.location.surah}:${e.location.ayah}`);
  const keep = (es: ClipScore["falsePositives"]) => es.filter((e) => !onLead(e));
  return {
    score: { ...score, final: keep(score.final), confirmed: keep(score.confirmed), falsePositives: keep(score.falsePositives) },
    leadInWarnings: score.falsePositives.filter(onLead).length,
  };
}
