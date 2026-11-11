// Stand-ins for the app's own UI code that the snippets call. Not part of @tilawa/core.
import type { ErrorClass, WordRef } from "@tilawa/core";

declare global {
  /** Moves the app's highlight to a word. */
  function highlight(word: WordRef): void;
  /** Marks a mistake in the app's UI. */
  function mark(location: WordRef, errorClass: ErrorClass): void;
  /** Shows a message to the reader. */
  function showMessage(text: string): void;
  /** A pending (suspected) mistake in the app's UI. */
  function showPending(id: string, location: WordRef): void;
  function clearPending(id: string): void;
  /** Audio the app captured itself, and its sample rate. */
  const chunks: Float32Array[];
  const sampleRate: number;
  const bar: HTMLProgressElement;
  const startButton: HTMLButtonElement;
  const stopButton: HTMLButtonElement;
}
export {};
