// docs/migration-0.x-to-1.0.md: the 1.0 side of each before/after pair.
import { ayahsRecited } from "@tilawa/core";
import type { EngineEvent } from "@tilawa/core";
import { createTilawa } from "@tilawa/core/web";

// Setup
const tilawa = await createTilawa(); // model + reference, downloaded once and cached
const session = tilawa.startFreeRecitation();
session.subscribe((e) => {
  if (e.type === "position" && e.state === "locked" && e.observed[0]) {
    const { surah, ayah, word } = e.observed[0].word;
    console.log(`${surah}:${ayah}, word ${word}`);
  }
});

// Mistakes
session.subscribe((e) => {
  if (e.type !== "checker") return;
  if (e.state === "suspected") showPending(e.id, e.location);
  if (e.state === "confirmed") mark(e.location, e.errorClass);
  if (e.state === "resolved" || e.state === "uncertain") clearPending(e.id);
});

// The final sequence
const events: EngineEvent[] = [];
session.subscribe((e) => events.push(e));

// Feeding audio and stopping
for (const chunk of chunks) session.pushAudio(chunk, sampleRate); // no need to await each one
await session.end();
const mistakes = session.mistakes(); // the confirmed ones
const recited = ayahsRecited(events); // [{ surah, ayah }, ...] in order
console.log(mistakes.length, recited);

// Next take: a new session. Sessions are one-shot.
session.dispose();
const next = tilawa.startFreeRecitation();
const mic = await tilawa.listen(next);
await mic.stop(); // stops the microphone, then ends the session

// A declared passage
const passage = tilawa.startMemorization({ surah: 18, fromAyah: 60, toAyah: 70 });
passage.dispose();

// Clean up the old 0.x model cache
indexedDB.deleteDatabase("tarteel-models");
