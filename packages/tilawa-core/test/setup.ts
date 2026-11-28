import { afterEach } from "vitest";

/**
 * The engine's tests are long and synchronous. A test worker reads its reporter's replies only when its event loop
 * polls, and waits 60 s for each before it reports `Timeout calling "onTaskUpdate"`: an unhandled error that fails the
 * run though every test passed. Tests run back to back without that poll, so a file whose tests together take longer
 * than that would fail the run. Each test ends by letting the loop poll once: an immediate queued from an immediate runs
 * in the loop's next turn, after its poll phase.
 */
afterEach(() => new Promise<void>((resolve) => setImmediate(() => setImmediate(resolve))));
