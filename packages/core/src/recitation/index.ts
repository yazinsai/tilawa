export { DEFAULT_CONFIG, BUFFER_CAP, FBANK_BINS, CTC_HZ } from "./config.js";
export type { EngineConfig } from "./config.js";
export { KaldiFbank } from "./fbank.js";
export {
  ZipformerRunner,
  defaultExecutionProviders,
  prepareOrtWasm,
} from "./zipformerRunner.js";
export type {
  ZipformerIo,
  ZipformerIoInput,
  OrtLike,
  OrtSessionLike,
  OrtWasmEnv,
  TensorLike,
} from "./zipformerRunner.js";
export { GreedyCtcDecoder, expandTokens } from "./ctcDecoder.js";
export { TOKENS, BLANK_ID, VOCAB_SIZE } from "./tokens.js";
export { QuranCorpus } from "./corpus.js";
export {
  CostTable,
  costTable,
  charCost,
  charId,
  ALPHABET,
  TABLE_SIZE,
  UNKNOWN_ID,
} from "./phonemeCost.js";
export {
  normalizedDistance,
  weightedLevenshtein,
  alignGlobal,
  alignSemiGlobal,
} from "./alignment.js";
export {
  QuranIndex,
  stripPreambles,
  fnv1aBucket,
  ISTIADHA,
  BASMALA,
} from "./search.js";
export { Tracker } from "./tracker.js";
export { VerdictTracer, pausalPhonemes } from "./verdicts.js";
export { RecitationEngine } from "./engine.js";
export { wholeAyahFallback } from "./fallback.js";
export type {
  CtcToken,
  HeardChar,
  VerdictState,
  WordVerdict,
  EngineEvent,
  EngineState,
  SearchHit,
  SearchResult,
  SearchHint,
  ExpectedPassage,
  FallbackHit,
  StripResult,
} from "./types.js";

// Verse emission policy: per-word verdicts -> ayah-level SDK events.
export {
  MIN_WORD_FRACTION,
  FALLBACK_MAX_DISTANCE,
  GAP_MAX_WORDS,
  ayahKey,
  ayahConfidence,
  ayahMeetsGate,
  snapshotTallies,
  accumulateSnapshot,
  mergeTallies,
  newlyEligibleAyahs,
  shouldRunFallback,
  fallbackConfidence,
  buildFinalSequence,
  bridgeGapAyahs,
  wordProgressFromCursor,
} from "./emission.js";
export type {
  AyahTally,
  BridgedAyahTally,
  EmissionVerdict,
  EmissionFallback,
  EmissionCursor,
  WordCountFn,
} from "./emission.js";

// The streaming session: PCM in, SDK verse events out.
export {
  ZipformerSession,
  createZipformerSession,
  displayQuranFromRaw,
  DEFAULT_ZIPFORMER_IO,
  AYAH_HEARD_FRACTION,
  DEFAULT_STRUCTURAL,
} from "./session.js";
export type {
  ZipformerSessionOptions,
  StructuralOptions,
  ModelSource,
  CorpusSource,
  QuranSource,
} from "./session.js";

export { CorrectionController, possibleWordIssues, DEFAULT_CORRECTION_THRESHOLDS, AYAH_ISSUE_KINDS } from "./correction.js";
export type { RecitationMode, CorrectionAction, CorrectionIssue, CorrectionState, CorrectionThresholds, RecitationPosition } from "./correction.js";
export {
  A0W_SLIP_HEAD,
  EncoderFrames,
  ENCODER_DIM,
  SLIP_PAD,
  poolEncoderSpan,
  slipHeadFromJson,
  slipProbability,
  slipThreshold,
  wordSpansFromTrail,
} from "./slipHead.js";
export type { SlipHead, SlipSensitivity, SlipWeightJson, TrailWordSpan } from "./slipHead.js";
export { vowelMismatches } from "./verdicts.js";
export { FramePosteriors, GOP_FLOOR, encodePhonemes, forcedLogLik, freeLogLik, pairScores, wordGop } from "./posteriors.js";
export type { WordGop } from "./posteriors.js";
export {
  StructuralRules,
  AYAH_ORDER_PARAMS,
  AYAH_ORDER_RULE,
  AYAH_ORDER_RULE_GUARDED,
  SIMILAR_VERSE_RULE,
  ayahOrderFlags,
  similarVerseEligible,
  similarVersePick,
  segmentTokens,
  pyRound,
} from "./structural.js";
export type {
  TimedToken,
  StructuralIndexJson,
  AyahOrderParams,
  AyahOrderRule,
  AyahOrderCandidate,
  AyahOrderJump,
  SimilarVerseRule,
  SimilarVerseCandidate,
  StructuralFlag,
} from "./structural.js";
