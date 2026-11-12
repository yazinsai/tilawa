// Engine event types as the report module carries them (contract §4), derived so /help does not depend on where
// the engine exports them from.
import type { TilawaReport } from "@tilawa/core/report";

export type EngineEvent = TilawaReport["observations"]["events"][number];
export type CheckerEvent = Extract<EngineEvent, { type: "checker" }>;
export type PositionEvent = Extract<EngineEvent, { type: "position" }>;
export type CapabilityEvent = Extract<EngineEvent, { type: "capabilities" }>;
export type WordRef = CheckerEvent["location"];
export type TrackerState = PositionEvent["state"];
