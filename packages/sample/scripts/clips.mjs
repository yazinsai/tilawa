// Links a fixed set of help eval clips into public/clips (gitignored) for the device page: the first 6 scored eval
// clips with a confirmed memorization error and the first 6 clean ones, by id. HELP_DIR defaults to the main checkout's.
import { existsSync, mkdirSync, readFileSync, symlinkSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const [root] = process.argv.slice(2);
const help = process.env.HELP_DIR ?? resolve(root, ".scratch/help");
const out = resolve("public/clips");
if (!existsSync(resolve(help, "labels.jsonl"))) {
  console.warn(`No help clips at ${help}; the device page will have no clips (set HELP_DIR)`);
  process.exit(0);
}
const MEMO = new Set(["wrong_word", "omitted_word", "added_word", "skipped_passage"]);
const labels = readFileSync(resolve(help, "labels.jsonl"), "utf8").trim().split("\n").map((s) => JSON.parse(s))
  .filter((l) => l.split === "eval" && l.scored).sort((a, b) => a.id.localeCompare(b.id));
const errors = labels.filter((l) => l.events.some((e) => MEMO.has(e.errorClass) && e.state === "confirmed")).slice(0, 6);
const clean = labels.filter((l) => l.kind === "none").slice(0, 6);
mkdirSync(out, { recursive: true });
const clips = [...errors, ...clean].map((l) => {
  const dst = resolve(out, `${l.id}.wav`);
  rmSync(dst, { force: true });
  symlinkSync(resolve(help, l.wav), dst);
  return { id: l.id, kind: l.kind, ayahs: l.ayahs, events: l.events.filter((e) => MEMO.has(e.errorClass)) };
});
writeFileSync(resolve(out, "clips.json"), JSON.stringify(clips, null, 1));
console.log(`${clips.length} clips in public/clips`);
