import type { AcousticTransitionModel } from "./acoustic-transition.js";
/** Installed only after speaker-separated training and premise validation. */
export const acousticTransitionModel: AcousticTransitionModel | null = null;

export const acousticTransitionEnabled = acousticTransitionModel !== null;
