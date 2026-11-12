import type { TrajectoryModel } from "./trajectory.js";

/** Populated only by a successful speaker-separated training run; never activate unfitted parameters. */
export const CONDITIONAL_TRAJECTORY_MODEL: TrajectoryModel | null = null;
