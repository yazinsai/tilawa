// docs/quickstart.md: "Node"
import { createTilawa, readWav } from "@tilawa/core/node";

const tilawa = await createTilawa(); // model cached in ~/.cache/tilawa; the reference ships in the package
const session = tilawa.startMemorization({ surah: 1, fromAyah: 1, toAyah: 7 });
session.subscribe((e) => {
  if (e.type === "checker" && e.state === "confirmed") console.log(e.errorClass, e.location);
});

const { samples, sampleRate } = await readWav("take.wav");
const chunk = Math.round(sampleRate * 0.3); // 300 ms at a time, like a live stream
for (let i = 0; i < samples.length; i += chunk) session.pushAudio(samples.subarray(i, i + chunk), sampleRate);
await session.end(); // waits for every queued push, then decides every open mistake

console.log(session.mistakes());
await tilawa.dispose();
