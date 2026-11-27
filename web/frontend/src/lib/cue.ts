/** A loud double beep (and a buzz on phones) when a word is flagged. On by default; the choice is remembered. */
const KEY = "tilawa:cue";
const GAP_MS = 400; // one cue for a burst of flags

/** The Android app (web/android) sounds the alert itself on the alarm stream, so it does not depend on the media volume. */
const app = (globalThis as { TilawaApp?: { alarm(): void } }).TilawaApp;
let ctx: AudioContext | null = null;
let last = -Infinity;
let on = (() => {
  try {
    return localStorage.getItem(KEY) !== "off";
  } catch {
    return true; // storage blocked: still cue, just don't remember
  }
})();

export const cueEnabled = () => on;

export function setCueEnabled(value: boolean) {
  on = value;
  try {
    localStorage.setItem(KEY, value ? "on" : "off");
  } catch {
    /* not remembered */
  }
}

/** Call inside a tap: iOS Safari only plays audio from a context created or resumed in a user gesture. */
export function armCue() {
  if (!on || typeof AudioContext === "undefined") return;
  ctx ??= new AudioContext();
  void ctx.resume().catch(() => {});
}

export function cue(now = performance.now()) {
  if (!on || now - last < GAP_MS) return;
  last = now;
  navigator.vibrate?.(200);
  if (app) return app.alarm();
  if (ctx?.state !== "running") return;
  const t = ctx.currentTime;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  // Meant to be heard across a room. The microphone runs without echo cancellation, so the engine hears it too.
  osc.type = "square";
  osc.frequency.setValueAtTime(1000, t);
  for (const [at, level] of [[0, 0.5], [0.15, 0], [0.22, 0.5], [0.37, 0]]) gain.gain.setValueAtTime(level, t + at);
  osc.connect(gain).connect(ctx.destination);
  osc.start(t);
  osc.stop(t + 0.37);
}
