// docs/quickstart.md: "A declared passage"
import { createTilawa } from "@tilawa/core/web";

const tilawa = await createTilawa();
// The reader is about to recite Al-Kahf 18:60-70 from memory.
const session = tilawa.startMemorization(
  { surah: 18, fromAyah: 60, toAyah: 70 },
  { confirm: { after: "next-word" } }, // confirm a mistake once the next word starts (default: 2.5 s after it)
);
session.subscribe((e) => {
  // In a memorization session, `expected` is the passage word the reader should be on.
  if (e.type === "position" && e.expected) highlight(e.expected);
  if (e.type === "checker" && e.state === "confirmed") mark(e.location, e.errorClass);
});
const mic = await tilawa.listen(session);
stopButton.onclick = () => void mic.stop();
