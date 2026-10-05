import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { join } from "node:path";
import { parseArgs } from "node:util";
import type { ReferenceAsset } from "../reference.js";
import type { ProposedLabel } from "../relabel/index.js";
import { STRATA, drawSample, latestAnswers, score } from "./index.js";
import type { ReviewAnswer, Sample, Score } from "./index.js";

const USAGE = `usage:
  review sample --labels <proposed-labels.jsonl> [--posterior-labels <proposed-labels.posteriors.jsonl>] --dir <dir> [--seed 1]
  review serve  --dir <dir> [--port 5177]
  review score  --dir <dir> --labels <proposed-labels.jsonl> [--posterior-labels <...>] [--target 0.98]`;

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    labels: { type: "string" },
    "posterior-labels": { type: "string" },
    dir: { type: "string" },
    seed: { type: "string", default: "1" },
    port: { type: "string", default: "5177" },
    target: { type: "string", default: "0.98" },
  },
});
const fail = (msg: string): never => {
  console.error(`${msg}\n\n${USAGE}`);
  process.exit(2);
};
const dir = values.dir ?? fail("--dir is required");
const readJsonl = <T>(path: string): T[] =>
  existsSync(path)
    ? readFileSync(path, "utf8")
        .split("\n")
        .filter(Boolean)
        .map((l) => JSON.parse(l))
    : [];
const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v1.json", import.meta.url), "utf8"));
const lastWord = (s: number, a: number) => asset.ayahs.find((x) => x.s === s && x.a === a)?.w.length ?? 0;
const samplePath = join(dir, "sample.json");
const answersPath = join(dir, "answers.jsonl");
const loadSample = (): Sample => (existsSync(samplePath) ? JSON.parse(readFileSync(samplePath, "utf8")) : fail(`no ${samplePath}; run "review sample" first`));

switch (positionals[0]) {
  case "sample": {
    const labels = readJsonl<ProposedLabel>(values.labels ?? fail("--labels is required"));
    const post = values["posterior-labels"] ? readJsonl<ProposedLabel>(values["posterior-labels"]) : undefined;
    if (existsSync(samplePath)) fail(`${samplePath} exists; a new sample would orphan its answers. Delete it first if you mean it.`);
    const sample = drawSample(labels, { seed: Number(values.seed), withPosteriors: post });
    mkdirSync(join(dir, "audio"), { recursive: true });
    writeFileSync(samplePath, JSON.stringify(sample, null, 2) + "\n");
    writeFileSync(join(dir, "rows.txt"), sample.clips.map((c) => c.row).join("\n") + "\n");
    console.table(Object.fromEntries(STRATA.map((s) => [s, sample.strata[s]])));
    console.log(`${sample.clips.length} clips -> ${samplePath}`);
    break;
  }

  case "serve": {
    const sample = loadSample();
    // The page only learns the surah, never the filename ayah or anything the tracker said.
    const surahOf = (path: string) => Number(/(\d+)_\d+_/.exec(path)?.[1] ?? 0);
    const surahs = [...new Set(sample.clips.map((c) => surahOf(c.path)))];
    const text = Object.fromEntries(surahs.map((s) => [s, asset.ayahs.filter((a) => a.s === s).map((a) => a.w)]));
    // Word counts for every ayah, so a span typed in for another surah still ends on a real word.
    const lengths: Record<number, number[]> = {};
    for (const a of asset.ayahs) (lengths[a.s] ??= [])[a.a - 1] = a.w.length;
    // Search ignores harakat, Quranic marks and every alef and hamza form: Uthmani writes many long "a" as a dagger
    // alef or not at all, so plain spellings of what you heard would otherwise miss.
    const bare = (t: string) =>
      t
        .replace(/[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u0640]/g, "")
        .replace(/[اٱأإآٰء]/g, "")
        .replace(/[ىۦئ]/g, "ي")
        .replace(/[ۥؤ]/g, "و")
        .replace(/ة/g, "ه")
        .replace(/\s+/g, " ")
        .trim();
    const searchable = asset.ayahs.map((a) => ({ s: a.s, a: a.a, text: a.w.join(" "), bare: bare(a.w.join(" ")) }));
    const page = readFileSync(new URL("./page.html", import.meta.url), "utf8");
    const server = createServer((req, res) => {
      const url = new URL(req.url ?? "/", "http://x");
      if (req.method === "GET" && url.pathname === "/") {
        res.writeHead(200, { "content-type": "text/html; charset=utf-8" }).end(page);
      } else if (req.method === "GET" && url.pathname === "/api/state") {
        const clips = sample.clips.map((c) => ({ shard: c.shard, row: c.row, surah: surahOf(c.path) }));
        const answers = Object.fromEntries(latestAnswers(readJsonl<ReviewAnswer>(answersPath)));
        res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ clips, answers, text, lengths }));
      } else if (req.method === "GET" && url.pathname.startsWith("/api/surah/")) {
        const s = Number(url.pathname.slice(11));
        res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(asset.ayahs.filter((a) => a.s === s).map((a) => a.w)));
      } else if (req.method === "GET" && url.pathname === "/api/search") {
        const q = bare(url.searchParams.get("q") ?? "");
        const hits = q ? searchable.filter((a) => a.bare.includes(q)).slice(0, 40).map(({ s, a, text }) => ({ s, a, text })) : [];
        res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(hits));
      } else if (req.method === "GET" && url.pathname.startsWith("/audio/")) {
        const file = join(dir, "audio", `${Number(url.pathname.slice(7))}.flac`);
        if (!existsSync(file)) res.writeHead(404).end("missing audio; run tools/extract_review_audio.py");
        else res.writeHead(200, { "content-type": "audio/flac" }).end(readFileSync(file));
      } else if (req.method === "POST" && url.pathname === "/api/answer") {
        let body = "";
        req.on("data", (d) => (body += d));
        req.on("end", () => {
          const a: ReviewAnswer = JSON.parse(body);
          appendFileSync(answersPath, JSON.stringify(a) + "\n");
          res.writeHead(204).end();
        });
      } else res.writeHead(404).end();
    });
    server.listen(Number(values.port), () => console.log(`review: http://localhost:${values.port}  (answers -> ${answersPath})`));
    break;
  }

  case "score": {
    const sample = loadSample();
    const answers = readJsonl<ReviewAnswer>(answersPath);
    const target = Number(values.target);
    const runs: [string, string][] = [["decoded", values.labels ?? fail("--labels is required")]];
    if (values["posterior-labels"]) runs.push(["posteriors", values["posterior-labels"]]);
    const scores = runs.map(([name, path]) => score(name, sample, readJsonl<ProposedLabel>(path), answers, lastWord));
    writeFileSync(join(dir, "score.json"), JSON.stringify(scores, null, 2) + "\n");
    const md = scores.map((s) => report(s, target)).join("\n\n");
    writeFileSync(join(dir, "score.md"), md + "\n");
    console.log(md);
    break;
  }

  default:
    fail(`unknown command "${positionals[0] ?? ""}"`);
}

function report(s: Score, target: number): string {
  const pct = (r: number | null) => (r === null ? "–" : `${(r * 100).toFixed(1)}%`);
  const ci = (c: [number, number]) => `${pct(c[0])}–${pct(c[1])}`;
  const cell = (r: { k: number; n: number; rate: number | null; ci: [number, number] }) => (r.n ? `${r.k}/${r.n} ${pct(r.rate)} (${ci(r.ci)})` : "–");
  const lines = [
    `## ${s.run}`,
    "",
    `Weighted to the whole run: precision ${pct(s.weighted.precision.rate)} (${ci(s.weighted.precision.ci)}), false locks ${pct(s.weighted.falseLock.rate)} (${ci(s.weighted.falseLock.ci)}), excluded ${pct(s.weighted.exclusion.rate)} (${ci(s.weighted.exclusion.ci)}).`,
    "",
    "| stratum | population | reviewed | precision (exact) | precision (ayahs) | false locks | excluded | excluded but clean | filename wrong |",
    "|---|---|---|---|---|---|---|---|---|",
    ...s.strata.map((x) => `| ${x.stratum} | ${x.population} | ${x.reviewed} | ${cell(x.precision)} | ${cell(x.ayahPrecision)} | ${cell(x.falseLock)} | ${cell(x.exclusion)} | ${cell(x.missed)} | ${cell(x.filenameWrong)} |`),
    "",
    "| min confidence | precision | 95% CI | share of run kept |",
    "|---|---|---|---|",
    ...s.sweep.map((x) => `| ${x.minConfidence} | ${pct(x.precision.rate)} | ${ci(x.precision.ci)} | ${pct(x.coverage)} |`),
  ];
  // The CI's lower bound, not the point estimate, has to clear the target.
  const pick = s.sweep.find((x) => x.precision.ci[0] >= target);
  lines.push("", pick ? `Lowest min confidence whose CI clears ${pct(target)}: **${pick.minConfidence}** (keeps ${pct(pick.coverage)} of the run).` : `No min confidence clears ${pct(target)} at the CI's lower bound yet.`);
  return lines.join("\n");
}
