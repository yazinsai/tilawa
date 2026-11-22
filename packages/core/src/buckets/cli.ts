import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "../reference.js";
import type { ReferenceAsset } from "../reference.js";
import type { ProposedLabel } from "../relabel/index.js";
import { BUCKETS, PROPOSED_CUTOFFS, SPLITS, bucketClip, buildManifests } from "./index.js";
import type { BucketCutoffs, ManifestClip } from "./index.js";
import { CUTOFF_QUOTA, CUTOFF_STRATA, drawCutoffSample, scoreCutoffs } from "./review.js";
import type { CutoffAnswer, CutoffClip, CutoffSample, CutoffScore, CutoffStratum } from "./review.js";

const USAGE = `usage:
  buckets build  --labels <dir of proposed-labels jsonl> --out <dir> --version <v> [--clusters <tsv>] [--exclude <stems.txt>] [--cutoffs <json>]
  buckets sample --clips <clips.jsonl> --dir <review dir> [--seed 1] [--quota text_only=30,...]
  buckets serve  --dir <review dir> [--port 5178]
  buckets score  --dir <review dir> [--target 0.95]`;

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    labels: { type: "string" },
    out: { type: "string" },
    version: { type: "string" },
    clusters: { type: "string" },
    exclude: { type: "string" },
    cutoffs: { type: "string" },
    clips: { type: "string" },
    dir: { type: "string" },
    seed: { type: "string", default: "1" },
    quota: { type: "string" },
    port: { type: "string", default: "5178" },
    target: { type: "string", default: "0.95" },
  },
});
const fail = (msg: string): never => {
  console.error(`${msg}\n\n${USAGE}`);
  process.exit(2);
};
const lines = (path: string) => readFileSync(path, "utf8").split("\n").filter(Boolean);
const readJsonl = <T>(path: string): T[] => (existsSync(path) ? lines(path).map((l) => JSON.parse(l)) : []);
const writeJsonl = (path: string, rows: unknown[]) => writeFileSync(path, rows.map((r) => JSON.stringify(r)).join("\n") + (rows.length ? "\n" : ""));
const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));

switch (positionals[0]) {
  case "build": {
    const labelsDir = values.labels ?? fail("--labels is required");
    const out = join(values.out ?? fail("--out is required"), values.version ?? fail("--version is required"));
    const cutoffs: BucketCutoffs = values.cutoffs ? { ...PROPOSED_CUTOFFS, ...JSON.parse(readFileSync(values.cutoffs, "utf8")) } : PROPOSED_CUTOFFS;
    const index = loadReferenceIndex(asset);
    const clusterOf = new Map<string, string>();
    if (values.clusters) for (const l of lines(values.clusters)) {
      const [shard, row, cluster] = l.split("\t");
      clusterOf.set(`${shard}:${row}`, cluster);
    }
    const excludeStems = new Set(values.exclude ? lines(values.exclude).map((s) => s.trim()) : []);
    const files = readdirSync(labelsDir).filter((f) => f.endsWith(".jsonl")).sort();
    const clips: (ManifestClip & CutoffClip)[] = [];
    for (const f of files)
      for (const l of readJsonl<ProposedLabel>(join(labelsDir, f))) {
        const b = bucketClip(index, l, l.teacher, cutoffs);
        clips.push({
          shard: l.shard,
          row: l.row,
          path: l.path,
          ...b,
          durationS: l.teacher?.durationS ?? null,
          cluster: clusterOf.get(`${l.shard}:${l.row}`) ?? null,
          per: l.per,
          meanConf: l.teacher?.meanConf ?? null,
          confidence: l.confidence,
        });
      }
    const fractions = { val: 0.05, test: 0.05 };
    const m = buildManifests(clips, { version: values.version!, excludeStems, fractions });
    for (const b of BUCKETS) mkdirSync(join(out, b), { recursive: true });
    for (const b of BUCKETS) for (const s of SPLITS) writeJsonl(join(out, b, `${s}.jsonl`), m.files[`${b}/${s}`] ?? []);
    writeJsonl(join(out, "clips.jsonl"), clips);
    const reasons: Record<string, number> = {};
    for (const c of clips) reasons[`${c.bucket}/${c.reason}`] = (reasons[`${c.bucket}/${c.reason}`] ?? 0) + 1;
    const unclustered = clips.filter((c) => c.cluster === null && c.bucket !== "reject").length;
    const manifest = {
      ...m.manifest,
      createdAt: new Date().toISOString(),
      referenceVersion: index.version,
      shards: files.length,
      clips: clips.length,
      cutoffs,
      cutoffsTuned: false,
      clusters: values.clusters ?? null,
      unclusteredNonReject: unclustered,
      reasons,
      textOnlyLabels: clips.filter((c) => c.label?.kind === "text").length,
    };
    writeFileSync(join(out, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
    writeFileSync(join(out, "distributions.json"), JSON.stringify(distributions(clips), null, 2) + "\n");
    console.log(JSON.stringify({ out, clips: clips.length, counts: m.manifest.counts, excluded: m.manifest.excluded, textOnly: manifest.textOnlyLabels, unclustered }, null, 2));
    break;
  }

  case "sample": {
    const dir = values.dir ?? fail("--dir is required");
    const path = join(dir, "sample.json");
    if (existsSync(path)) fail(`${path} exists; a new sample would orphan its answers. Delete it first if you mean it.`);
    const clips = readJsonl<CutoffClip>(values.clips ?? fail("--clips is required"));
    const quota = { ...CUTOFF_QUOTA };
    for (const kv of (values.quota ?? "").split(",").filter(Boolean)) {
      const [k, v] = kv.split("=");
      if (!(CUTOFF_STRATA as readonly string[]).includes(k) || !(Number(v) >= 0)) fail(`bad --quota entry "${kv}"`);
      quota[k as CutoffStratum] = Number(v);
    }
    const sample = drawCutoffSample(clips, PROPOSED_CUTOFFS, { seed: Number(values.seed), quota });
    mkdirSync(join(dir, "audio"), { recursive: true });
    writeFileSync(path, JSON.stringify(sample, null, 2) + "\n");
    writeFileSync(join(dir, "rows.tsv"), sample.clips.map((c) => `${c.shard}\t${c.row}`).join("\n") + "\n");
    console.table(sample.strata);
    console.log(`${sample.clips.length} clips -> ${path}`);
    break;
  }

  case "serve": {
    const dir = values.dir ?? fail("--dir is required");
    const sample: CutoffSample = JSON.parse(readFileSync(join(dir, "sample.json"), "utf8"));
    const answersPath = join(dir, "answers.jsonl");
    const words = new Map(asset.ayahs.map((a) => [`${a.s}:${a.a}`, a.w]));
    const textOf = (c: CutoffClip): { heading: string; text: string } | null => {
      const span = c.label?.kind === "span" ? c.label.span : c.label?.kind === "text" ? c.label.occurrences[0] : null;
      if (!span) return null;
      const out: string[] = [];
      for (let s = span.start.surah, a = span.start.ayah; ; ) {
        const w = words.get(`${s}:${a}`) ?? [];
        const from = s === span.start.surah && a === span.start.ayah ? span.start.word : 1;
        const to = s === span.end.surah && a === span.end.ayah ? span.end.word : w.length;
        out.push(w.slice(from - 1, to).join(" ") + ` ﴿${a}﴾`);
        if (s === span.end.surah && a === span.end.ayah) break;
        if (words.has(`${s}:${a + 1}`)) a++;
        else (s++, (a = 1));
        if (out.length > 40) break;
      }
      const where = `${span.start.surah}:${span.start.ayah}:${span.start.word} – ${span.end.surah}:${span.end.ayah}:${span.end.word}`;
      return { heading: c.label?.kind === "text" ? `text only (${c.label.occurrences.length}+ occurrences, e.g. ${where})` : where, text: out.join(" ") };
    };
    // Unlocated clips show the tracker's best guess so "wrong passage" vs "no recitation" can still be told apart.
    const page = readFileSync(new URL("./page.html", import.meta.url), "utf8");
    const server = createServer((req, res) => {
      const url = new URL(req.url ?? "/", "http://x");
      if (req.method === "GET" && url.pathname === "/") res.writeHead(200, { "content-type": "text/html; charset=utf-8" }).end(page);
      else if (req.method === "GET" && url.pathname === "/api/state") {
        const clips = sample.clips.map((c) => ({ shard: c.shard, row: c.row, label: textOf(c) }));
        const answers = Object.fromEntries(readJsonl<CutoffAnswer>(answersPath).map((a) => [`${a.shard}:${a.row}`, a]));
        res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ clips, answers }));
      } else if (req.method === "GET" && url.pathname.startsWith("/audio/")) {
        const file = join(dir, "audio", `${decodeURIComponent(url.pathname.slice(7))}.flac`);
        if (!existsSync(file)) res.writeHead(404).end("missing audio; run tools/extract_review_audio.py --pack");
        else res.writeHead(200, { "content-type": "audio/flac" }).end(readFileSync(file));
      } else if (req.method === "POST" && url.pathname === "/api/answer") {
        let body = "";
        req.on("data", (d) => (body += d));
        req.on("end", () => {
          appendFileSync(answersPath, JSON.stringify(JSON.parse(body)) + "\n");
          res.writeHead(204).end();
        });
      } else res.writeHead(404).end();
    });
    server.listen(Number(values.port), () => console.log(`bucket review: http://localhost:${values.port}  (answers -> ${answersPath})`));
    break;
  }

  case "score": {
    const dir = values.dir ?? fail("--dir is required");
    const sample: CutoffSample = JSON.parse(readFileSync(join(dir, "sample.json"), "utf8"));
    const s = scoreCutoffs(sample, readJsonl<CutoffAnswer>(join(dir, "answers.jsonl")), {
      cleanMaxPer: [0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
      cleanMinTeacherConf: [0.8, 0.85, 0.9, 0.93, 0.95],
    });
    writeFileSync(join(dir, "score.json"), JSON.stringify(s, null, 2) + "\n");
    const md = report(s, Number(values.target));
    writeFileSync(join(dir, "score.md"), md + "\n");
    console.log(md);
    break;
  }

  default:
    fail(`unknown command "${positionals[0] ?? ""}"`);
}

function distributions(clips: CutoffClip[]) {
  const located = clips.filter((c) => c.label !== null || c.reason === "high_per");
  const hist = (xs: number[], edges: number[]) => edges.slice(0, -1).map((lo, i) => ({ lo, hi: edges[i + 1], n: xs.filter((x) => x >= lo && x < edges[i + 1]).length }));
  const q = (xs: number[], ps: number[]) => {
    const s = [...xs].sort((a, b) => a - b);
    return Object.fromEntries(ps.map((p) => [`p${p}`, s.length ? s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))] : null]));
  };
  const per = located.map((c) => c.per ?? Infinity).filter(Number.isFinite);
  const conf = clips.map((c) => c.meanConf).filter((x): x is number => x !== null);
  return {
    locatedPer: { quantiles: q(per, [10, 25, 50, 75, 90, 95]), hist: hist(per, [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 1, Infinity]) },
    teacherMeanConf: { quantiles: q(conf, [5, 10, 25, 50, 75, 90]), hist: hist(conf, [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 1.01]) },
  };
}

function report(s: CutoffScore, target: number): string {
  const pct = (r: number | null) => (r === null ? "–" : `${(r * 100).toFixed(1)}%`);
  const out = [
    `# Bucket cutoff review`,
    "",
    `${s.answered} answered, ${s.unsure} unsure (left out).`,
    "",
    "| stratum | population | exact | mistakes | wrong passage | no recitation | unsure |",
    "|---|---|---|---|---|---|---|",
    ...Object.entries(s.strata).map(([k, v]) => `| ${k} | ${v!.population} | ${v!.verdicts.exact ?? 0} | ${v!.verdicts.mistakes ?? 0} | ${v!.verdicts.wrong_passage ?? 0} | ${v!.verdicts.no_recitation ?? 0} | ${v!.verdicts.unsure ?? 0} |`),
    "",
    "| clean max PER | min teacher conf | clean precision (exact) | 95% CI | clean share | deviant right passage | deviant with mistakes | deviant share |",
    "|---|---|---|---|---|---|---|---|",
    ...s.grid.map((g) => `| ${g.cleanMaxPer} | ${g.cleanMinTeacherConf} | ${pct(g.clean.precision.rate)} (${g.clean.precision.k}/${g.clean.precision.n}) | ${pct(g.clean.precision.ci[0])}–${pct(g.clean.precision.ci[1])} | ${pct(g.clean.share)} | ${pct(g.deviant.rightPassage.rate)} | ${pct(g.deviant.mistakes.rate)} | ${pct(g.deviant.share)} |`),
  ];
  const ok = s.grid.filter((g) => (g.clean.precision.rate ?? 0) >= target).sort((a, b) => b.clean.share - a.clean.share)[0];
  out.push("", ok ? `Largest clean-match with precision ≥ ${pct(target)}: **cleanMaxPer ${ok.cleanMaxPer}, cleanMinTeacherConf ${ok.cleanMinTeacherConf}** (${pct(ok.clean.share)} of the run).` : `No grid point reaches ${pct(target)} clean precision.`);
  return out.join("\n");
}
