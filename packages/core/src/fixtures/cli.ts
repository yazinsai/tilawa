import { readFileSync } from "node:fs";
import { align } from "../align.js";
import type { AlignmentOp } from "../align.js";
import { loadReferenceIndex } from "../reference.js";
import type { ReferenceAsset } from "../reference.js";
import { diffExpectation, finalPosition, formatSpan, isAssertable, replay } from "./index.js";
import type { Fixture } from "./index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);

const renderPath = (path: AlignmentOp[]) =>
  path
    .map((o) => (o.op === "match" ? o.ref : o.op === "sub" ? `[${o.ref}→${o.obs}]` : o.op === "del" ? `[-${o.ref}]` : `[+${o.obs}]`))
    .join("");

const args = process.argv.slice(2);
const files = args.filter((a) => !a.startsWith("--"));

if (args.includes("--compare-posteriors")) {
  // Same fixtures, decoded only vs with posteriors, one line per fixture.
  let changed = 0;
  const describe = (e: ReturnType<typeof finalPosition>) =>
    e ? `${e.state} ${e.observed[0] ? `${formatSpan(e.observed[0].span)} conf=${e.observed[0].confidence.toFixed(2)}` : "-"}` : "none";
  for (const file of files) {
    const fixture: Fixture = JSON.parse(readFileSync(file, "utf8"));
    if (!fixture.evidence.some((e) => e.posteriors)) {
      console.log(`${fixture.id}  no posteriors`);
      continue;
    }
    const decoded = describe(finalPosition(replay(index, { ...fixture, evidence: fixture.evidence.map(({ posteriors: _, ...e }) => e) })));
    const reranked = describe(finalPosition(replay(index, fixture)));
    if (decoded !== reranked) changed++;
    console.log(`${decoded === reranked ? " " : "*"} ${fixture.id}  decoded: ${decoded}   posteriors: ${reranked}`);
  }
  console.log(`\n${changed} of ${files.length} fixtures differ`);
  process.exit(0);
}

let failed = 0;
for (const file of files) {
  const fixture: Fixture = JSON.parse(readFileSync(file, "utf8"));
  const events = replay(index, fixture, { trace: true });
  const final = finalPosition(events);
  const p = fixture.provenance;
  const label = p.kind === "tlog" ? `tlog ${p.path} (filename ${p.filenameLabel.surah}:${p.filenameLabel.ayah}, audio ${p.audioVerified ? "verified" : "unverified"})` : "synthetic";

  console.log(`\n${fixture.id}  [${label}]`);
  console.log(`  state: ${final?.state ?? "none"}   events: ${events.length}`);
  for (const c of final?.trace?.shortlist.slice(0, 5) ?? []) {
    console.log(`  candidate ${formatSpan(c.span)}  votes=${c.votes} cost=${c.cost.toFixed(3)}`);
  }
  if (final?.trace?.path) console.log(`  path: ${renderPath(final.trace.path)}`);
  else {
    const observed = fixture.evidence.flatMap((e) => e.phonemes.map((x) => x.p)).join("");
    console.log(`  observed: ${observed}`);
    if (p.kind === "tlog") {
      const ayah = asset.ayahs.find((a) => a.s === p.filenameLabel.surah && a.a === p.filenameLabel.ayah);
      if (ayah) console.log(`  vs filename ayah: ${renderPath(align(ayah.p, observed))}`);
    }
  }

  const problems = diffExpectation(fixture, events);
  if (fixture.expect) {
    const mode = isAssertable(fixture) ? "asserted" : "snapshot only";
    console.log(`  expected (${mode}): ${fixture.expect.state}${fixture.expect.span ? ` ${formatSpan(fixture.expect.span)}` : ""}`);
    for (const pr of problems) console.log(`  ✗ ${pr}`);
    if (problems.length && isAssertable(fixture)) failed++;
  }
}
process.exitCode = failed ? 1 : 0;
