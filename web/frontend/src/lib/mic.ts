import type { PreparedMicrophone } from "@tilawa/core/web";

/**
 * `prepareMicrophone()` from @tilawa/core/web, available before the SDK chunk loads. Call it synchronously in the
 * Start tap: iOS Safari only runs audio created or resumed inside the gesture, and the first download can take long
 * enough to lose it. Null without a microphone API; `listen()` then reports why. Same constraints as the SDK.
 */
export function prepareMic(): PreparedMicrophone | null {
  if (typeof AudioContext === "undefined" || !navigator.mediaDevices?.getUserMedia) return null;
  const context = new AudioContext();
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
