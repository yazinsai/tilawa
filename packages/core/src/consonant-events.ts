const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");
const VOWELS = new Set("اۦۥَُِ۪ٲ");
/** One label per articulation. Remove duration copies before vowels, retaining
 * vowel-separated and word-separated repetitions as distinct CTC events. */
export function consonantEvents(parts: readonly string[]): string {
  return parts.map(part => {
    let previous = "", result = "";
    for (const raw of part) {
      const p = raw === "ں" ? "ن" : raw === "۾" ? "م" : raw;
      if (CONSONANTS.has(p)) { if (p !== previous) result += p; previous = p; }
      else if (VOWELS.has(p)) previous = "";
    }
    return result;
  }).join("");
}
