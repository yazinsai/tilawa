// docs/self-hosting.md: "Mirror tilawa.dev/v1 with its manifest". Node 20+.
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

interface AssetManifest {
  schema: 1;
  ort: string;
  files: { path: string; bytes: number; sha256: string }[];
}

const source = "https://tilawa.dev/v1/";
const out = "public/tilawa/v1"; // served by your app as /tilawa/v1/

const manifest = (await (await fetch(source + "manifest.json")).json()) as AssetManifest;
for (const file of manifest.files) {
  const res = await fetch(source + file.path);
  if (!res.ok) throw new Error(`${res.status} fetching ${file.path}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (bytes.length !== file.bytes || sha256 !== file.sha256) throw new Error(`Checksum mismatch for ${file.path}`);
  await mkdir(dirname(join(out, file.path)), { recursive: true });
  await writeFile(join(out, file.path), bytes);
}
await writeFile(join(out, "manifest.json"), JSON.stringify(manifest, null, 1));
console.log(`Mirrored ${manifest.files.length} files (onnxruntime-web ${manifest.ort})`);
