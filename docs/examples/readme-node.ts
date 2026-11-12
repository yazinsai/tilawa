// The Node snippet in the root README.md, kept here so it is typechecked too.
import { createTilawa, readWav } from "@tilawa/core/node";

const tilawa = await createTilawa(); // model cached in ~/.cache/tilawa; the reference ships in the package
const session = tilawa.startMemorization({ surah: 1, fromAyah: 1, toAyah: 7 });
session.subscribe((e) => {
  if (e.type === "checker" && e.state === "confirmed") console.log(e.errorClass, e.location);
});

const { samples, sampleRate } = await readWav("take.wav");
for (let i = 0; i < samples.length; i += 4800) session.pushAudio(samples.subarray(i, i + 4800), sampleRate);
await session.end();
console.log(session.mistakes());
await tilawa.dispose();
