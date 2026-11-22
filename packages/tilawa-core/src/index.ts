/**
 * `@tilawa/core`: the runtime-neutral engine. Platform entries (`/web`, `/node`, `/react-native`) add `createTilawa`.
 * Curated: everything public is named here; the vendored research modules stay internal.
 */

// Engine and sessions
export { createEngine } from "./session.js";
export type {
  AudioTap,
  CapabilityEvent,
  DeclaredPassage,
  Engine,
  EngineEvent,
  EngineOptions,
  MemorizationPassage,
  ResolvedSessionPolicy,
  Session,
  SessionPolicy,
  SessionState,
  TilawaSession,
} from "./session.js";
export type { Microphone, MicrophoneOptions, LoadProgress, Tilawa, TilawaModelInfo, TilawaOptions } from "./tilawa.js";
export type { ListenerOptions } from "./runtime/listener.js";
export type { VadOptions } from "./runtime/audio.js";

// Events (schema v1)
export { EVENT_SCHEMA_VERSION } from "./engine/index.js";
export type {
  CheckerEvent,
  CheckerState,
  ErrorClass,
  MemorizationErrorClass,
  PositionCandidate,
  PositionEvent,
  PronunciationErrorClass,
  TrackerState,
  WordRef,
  WordSpan,
} from "./engine/index.js";
export { ALL_CLASSES, MEMORIZATION_CLASSES } from "./runtime/capabilities.js";
export { ayahsRecited } from "./recited.js";

// Evidence (pushEvidence / evidence())
export type { DecodedPhoneme, PhonemeEvidence, TopKPosteriors } from "./engine/index.js";

// Reference
export { loadReferenceIndex } from "./engine/index.js";
export type { AyahRef, PhraseAsset, ReferenceAsset, ReferenceIndex } from "./engine/index.js";

// Models
export { MODEL_REGISTRY, withBase } from "./runtime/registry.js";
export { MODEL_BUDGET_BYTES, fetchModel, memoryStorage } from "./runtime/models.js";
export type { FetchModelOptions, FetchedModel, ModelRegistry, ModelStorage, ModelTier, TierName } from "./runtime/models.js";
export { ortBackend } from "./runtime/ort.js";
export type { OrtModule } from "./runtime/ort.js";
export type { InferenceBackend, LoadedModel, ModelManifest, StateSpec, Tensor } from "./runtime/model.js";

// Defaults, version, errors
export { DEFAULTS, REFERENCE } from "./defaults.js";
export { SDK_VERSION } from "./version.js";
export { MicrophoneError, ModelBudgetError, ModelIntegrityError, SessionEndedError, TilawaLoadError } from "./errors.js";
export type { LoadStage, MicrophoneErrorReason } from "./errors.js";
