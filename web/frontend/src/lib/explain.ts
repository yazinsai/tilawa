import type { CheckerEvent, MemorizationErrorClass } from "@tilawa/core";

/** Plain-language wording for the four memorization classes V1 checks. Other classes aren't shown. */
export const EXPLAIN: Record<MemorizationErrorClass, { label: string; text: string }> = {
  wrong_word: { label: "Different word", text: "It sounded like a different word here." },
  omitted_word: { label: "Missed word", text: "This word seems to have been left out." },
  added_word: { label: "Extra word", text: "An extra word seems to have been added here." },
  skipped_passage: { label: "Skipped part", text: "Part of the passage seems to have been skipped from here." },
};

export const isShown = (e: CheckerEvent): e is CheckerEvent & { errorClass: MemorizationErrorClass } =>
  e.errorClass in EXPLAIN;
