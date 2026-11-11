// The declared-passage snippet in the root README.md.
import { createTilawa } from "@tilawa/core/web";

const tilawa = await createTilawa();
const session = tilawa.startMemorization(
  { surah: 18, fromAyah: 60, toAyah: 70 },
  { confirm: { after: "next-word" } }, // confirm a mistake once the next word starts (default: 2.5 s after it)
);
session.dispose();
