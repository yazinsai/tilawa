import type { TeacherObservationModel } from "./teacher-observation.js";

/** Installed only after speaker-separated training and control selection; never a synthetic fallback. */
export const TEACHER_OBSERVATION_MODEL: { mediated: TeacherObservationModel; direct: TeacherObservationModel } | null = null;
