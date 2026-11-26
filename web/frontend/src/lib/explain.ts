import type { CheckerEvent, MemorizationErrorClass } from "@tilawa/core";
import { tr } from "./i18n";

/** Plain-language wording for the four memorization classes V1 checks. Other classes aren't shown. */
export const EXPLAIN: Record<MemorizationErrorClass, { label: string; text: string }> = {
  wrong_word: { label: tr("wrongLabel"), text: tr("wrongText") },
  omitted_word: { label: tr("omittedLabel"), text: tr("omittedText") },
  added_word: { label: tr("addedLabel"), text: tr("addedText") },
  skipped_passage: { label: tr("skippedLabel"), text: tr("skippedText") },
};

export const isShown = (e: CheckerEvent): e is CheckerEvent & { errorClass: MemorizationErrorClass } =>
  e.errorClass in EXPLAIN;
