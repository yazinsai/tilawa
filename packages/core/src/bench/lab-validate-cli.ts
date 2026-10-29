/** Runs `validateSession` over every lab test set session (`<dir>/sessions/*.json`) and the manifest's inline copies; exits 1 on any failure. */
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import { validateSession } from "./sessions.js";
import type { BenchSession } from "./sessions.js";

const { values } = parseArgs({ options: { dir: { type: "string", default: `${process.env.LAB_DATA ?? "."}/lab-v1.3` }, manifest: { type: "string", default: "manifest-v1.json" },
  out: { type: "string", default: process.env.LAB_OUT ? `${process.env.LAB_OUT}/validate.json` : "" } } });
const asset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const dir = resolve(values.dir!);
const failures: { file: string; error: string }[] = [];
const check = (file: string, s: BenchSession) => {
  try { validateSession(s, index); } catch (e) { failures.push({ file, error: (e as Error).message }); }
  for (const e of s.events) if ((e.errorClass === "wrong_word" || e.errorClass === "added_word") && !e.said)
    failures.push({ file, error: `planted ${e.errorClass} at ${e.location.surah}:${e.location.ayah}:${e.location.word} lacks said phonemes` });
};
const files = readdirSync(join(dir, "sessions")).filter((f) => f.endsWith(".json") && !/\.(checks|evidence)\.json$/.test(f)).sort();
for (const f of files) check(`sessions/${f}`, JSON.parse(readFileSync(join(dir, "sessions", f), "utf8")));
const manifestPath = join(dir, values.manifest!);
let inline = 0;
if (existsSync(manifestPath)) {
  const m = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (m.version !== 1 || !Array.isArray(m.sessions)) failures.push({ file: values.manifest!, error: "Invalid manifest" });
  for (const x of m.sessions ?? []) {
    check(`${values.manifest}#${x.id}`, x.session);
    if (!existsSync(resolve(dir, x.session.audio))) failures.push({ file: `${values.manifest}#${x.id}`, error: `missing audio ${x.session.audio}` });
    inline++;
  }
}
const result = { dir, files: files.length, manifestSessions: inline, failures };
if (values.out) writeFileSync(values.out, JSON.stringify(result, null, 1));
console.log(JSON.stringify(result, null, 1));
if (failures.length || !files.length) process.exit(1);
