import type { CheckerPolicy } from "./engine/index.js";
import type { Refinements } from "./runtime/session.js";

/** The reference this release's models are built against: `<assetBase>reference/ref-v2.json`, also in `assets/`. */
export const REFERENCE = Object.freeze({
  version: "ref-v2",
  path: "reference/ref-v2.json",
  bytes: 4_749_281,
  sha256: "f236384ea91e81ad554ba2777bc4073c3ff41e34120343a8a474571c577b7727",
});

/** What `createTilawa` and sessions use when an option is left out. */
export const DEFAULTS = Object.freeze({
  /** Models at `models/<id>/<version>/model.onnx`, the reference at `reference/ref-v2.json`, ORT wasm at `runtime/ort/<ortVersion>/`. */
  assetBase: "https://tilawa.dev/v1/",
  tier: "auto" as "full" | "lite" | "auto",
  /** Cache API cache (web) holding model and reference bytes. */
  cacheName: "tilawa-models",
  /** `listen()` gates audio with the VAD only when asked. */
  vad: false,
  policy: Object.freeze({
    countSelfCorrections: false,
    confirm: Object.freeze({ after: "delay", ms: 2500 }) as Readonly<CheckerPolicy["confirm"] & object>,
    minConfidence: 0.6,
    /**
     * The refinements of rc.1's rules (`Refinements`): every one on but `markWhenSure`, which stays off until it can tell
     * apart the cases it now gets wrong (see the field). Set a field false to restore its rc.1 rule, or set
     * `markWhenSure` true to turn it on.
     */
    refinements: Object.freeze({
      readNotSkipped: true,
      reanchorBack: true,
      stutter: true,
      ledgerPartial: true,
      omissionDue: true,
      commitAfterJump: true,
      firstAyahOut: true,
      marksInPassage: true,
      repeatNotWrong: true,
      markWhenSure: false,
      confirmAdvance: true,
      onsetRun: true,
      declaredStart: true,
      declaredLeadIn: true,
      declaredMove: true,
      noLostBeforeLock: true,
      consonantConfirm: true,
      rivalHold: true,
      passageWindow: true,
      restartNoEntry: true,
      withdrawBegun: true,
      leftOnNewAudio: true,
      begunByConsonants: true,
      showSaid: true,
      pace: true,
    }) as Readonly<Required<Refinements>>,
  }),
});
