import type { PreparedMicrophone } from "@tilawa/core/web";

/**
 * `prepareMicrophone()` from @tilawa/core/web, available before the SDK chunk loads. Call it synchronously in the
 * Start tap: iOS Safari only runs audio created or resumed inside the gesture, and the first download can take long
 * enough to lose it. Null without a microphone API; `listen()` then reports why. Same constraints as the SDK.
 * With `sampleRate` 16000 the browser resamples the microphone, and the SDK's JavaScript resampler has nothing to do.
 */
export function prepareMic(sampleRate?: number): PreparedMicrophone | null {
  if (typeof AudioContext === "undefined" || !navigator.mediaDevices?.getUserMedia) return null;
  const context = new AudioContext(sampleRate ? { sampleRate } : undefined);
  void context.resume().catch(() => {});
  const stream = navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  stream.catch(() => {});
  let released = false;
  return {
    context,
    stream,
    release() {
      if (released) return;
      released = true;
      void stream.then((s) => s.getTracks().forEach((t) => t.stop()), () => {});
      void context.close().catch(() => {});
    },
  };
}
