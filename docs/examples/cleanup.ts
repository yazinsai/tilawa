// docs/quickstart.md: "Stopping and cleaning up"
import { createTilawa } from "@tilawa/core/web";

// Cancel a slow first download, for example when the reader leaves the page.
const abort = new AbortController();
window.addEventListener("pagehide", () => abort.abort(), { once: true });
const tilawa = await createTilawa({ signal: abort.signal }); // rejects with an AbortError if cancelled

// One session is one take.
const session = tilawa.startFreeRecitation();
const unsubscribe = session.subscribe((e) => console.log(e.type));
const mic = await tilawa.listen(session);

// Stop: the microphone stops, then the session ends and its last events arrive.
await mic.stop();
const mistakes = session.mistakes(); // read results before disposing
unsubscribe();
session.dispose(); // drops subscribers and buffers; the model stays loaded

// Another take: start a new session. Sessions can't be restarted.
const next = tilawa.startFreeRecitation();
next.dispose();

// Done with Tilawa altogether: disposes any open sessions and releases the model.
await tilawa.dispose();
console.log(mistakes.length);
