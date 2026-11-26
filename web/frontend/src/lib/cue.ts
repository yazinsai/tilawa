/** A short, quiet tone (and a buzz on phones) when a word is flagged. On by default; the choice is remembered. */
const KEY = "tilawa:cue";
const GAP_MS = 400; // one cue for a burst of flags

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
  navigator.vibrate?.(60);
  if (ctx?.state !== "running") return;
  const t = ctx.currentTime;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.frequency.setValueAtTime(440, t);
  osc.frequency.exponentialRampToValueAtTime(330, t + 0.16);
  // Quiet and short: the microphone runs without echo cancellation, so the engine hears this too.
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(0.08, t + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.18);
  osc.connect(gain).connect(ctx.destination);
  osc.start(t);
  osc.stop(t + 0.2);
}
