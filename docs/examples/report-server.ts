// docs/reporting.md: "Run your own endpoint". Node 20+, no framework.
import { randomBytes } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { validateReport } from "@tilawa/core/report";

createServer(async (req, res) => {
  if (req.method !== "POST" || req.url !== "/tilawa") return void res.writeHead(404).end();
  // Parse the multipart body with the platform's Request/FormData.
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  const form = await new Request("http://local/", {
    method: "POST",
    headers: { "content-type": req.headers["content-type"] ?? "" },
    body: Buffer.concat(chunks),
  }).formData();
  const report = form.get("report");
  const audio = form.get("audio");
  if (!(report instanceof Blob) || !(audio instanceof Blob)) return void res.writeHead(400).end();

  const json = new Uint8Array(await report.arrayBuffer());
  const wav = new Uint8Array(await audio.arrayBuffer());
  const result = await validateReport(json, wav);
  if (!result.ok || !result.report) {
    res.writeHead(400, { "content-type": "application/json" });
    return void res.end(JSON.stringify({ errors: result.errors }));
  }
  // A real server must also: check consent.version against the versions you accept, refuse a reportId it already
  // has with different bytes (409), rate-limit, and store privately.
  const dir = `reports/${result.report.reportId}`;
  await mkdir(dir, { recursive: true });
  await writeFile(`${dir}/report.json`, json);
  await writeFile(`${dir}/audio.wav`, wav);
  res.writeHead(201, { "content-type": "application/json" });
  res.end(JSON.stringify({
    reportId: result.report.reportId,
    receivedAt: new Date().toISOString(),
    deleteToken: randomBytes(32).toString("base64url"), // store only its hash
  }));
}).listen(8787);
