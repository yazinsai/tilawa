/** Writes the per-word disambiguation table (`words[i]`: further words needed from word i, -1 never) for the lab session builder. */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import { disambiguationWords } from "../disambiguation.js";

const { values } = parseArgs({ options: { out: { type: "string", default: `${process.env.LAB_DATA ?? "."}/reference/disambiguation-v1.json` } } });
const asset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const t0 = performance.now();
const k = disambiguationWords(index);
const ref = (i: number) => `${index.words[i].surah}:${index.words[i].ayah}:${index.words[i].word}`;
const counts = new Map<string, number>();
for (const x of k) { const b = x < 0 ? "never" : x <= 5 ? String(x) : x <= 10 ? "6-10" : x <= 20 ? "11-20" : ">20"; counts.set(b, (counts.get(b) ?? 0) + 1); }
const longest = [...k.keys()].filter((i) => k[i] > 10 && (i === 0 || k[i - 1] !== k[i] + 1)).sort((a, b) => k[b] - k[a]).slice(0, 25);
const out = resolve(values.out!);
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify({ version: 1, reference: index.version, rule: "exact phoneme uniqueness over the whole Quran from each start word (packages/core/src/disambiguation.ts)", words: Array.from(k) }));
console.log(`${k.length} words in ${((performance.now() - t0) / 1000).toFixed(1)} s -> ${out}`);
console.log("further words needed:", Object.fromEntries([...counts].sort()));
console.log("longest shared runs (start: words):", longest.map((i) => `${ref(i)}: ${k[i]}`).join(", "));
