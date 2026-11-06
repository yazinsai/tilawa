import type { PhonemeEvidence } from "@tilawa/core";

/** Required on the decode path: a side-ear certificate must leave every primary byte unchanged. */
export function assertPrimaryEvidence(primary: PhonemeEvidence[], enriched: PhonemeEvidence[]): void {
  const strip = (chunks: PhonemeEvidence[]) => chunks.map(e => ({ ...e,
    phonemes: e.phonemes.map(({ wrongWordAgreements: _, ...p }) => p) }));
  if (JSON.stringify(strip(primary)) !== JSON.stringify(strip(enriched)))
    throw new Error("Pause-ear agreement changed primary evidence or folded teacher certificates");
}
