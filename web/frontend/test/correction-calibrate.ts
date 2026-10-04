/** Correction-mode calibration report: false-flag rate + per-word verdict fields on clean clips.
 * Run from web/frontend:
 *   node --import tsx test/correction-calibrate.ts [--clips <dir|file>] [--json out.json] [--verbose]
 * Default clip set = lab/benchmark/test_corpus/manifest.json (all correct recitations).
 * Exit code is always 0; this is a report, not a gate. */
import { readFileSync, existsSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { resolve, basename, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ZipformerSession, type WorkerOutbound } from '@tilawa/core';

const args = process.argv.slice(2);
const opt = (k: string) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : undefined; };
const verbose = args.includes('--verbose');
const jsonOut = opt('--json');
const clipsArg = opt('--clips');

const frontend = fileURLToPath(new URL('..', import.meta.url));
const repo = resolve(frontend, '../..');
const root = process.env.TILAWA_AUDIO_ROOT ?? [repo, resolve(repo, '../..')].find(p => existsSync(resolve(p, 'lab/benchmark/test_corpus/manifest.json')))!;
const corpusDir = resolve(root, 'lab/benchmark/test_corpus');

interface Ref { surah: number; ayah: number }
interface Clip { file: string; path: string; expected: Ref[]; alsoAccept: Ref[][] }
const clips: Clip[] = [];
if (clipsArg) {
  const p = resolve(clipsArg);
  const files = statSync(p).isDirectory()
    ? readdirSync(p).filter(f => /\.(wav|mp3|m4a)$/i.test(f)).sort().map(f => resolve(p, f)) : [p];
  for (const f of files) clips.push({ file: `${basename(resolve(f, '..'))}/${basename(f)}`, path: f, expected: [], alsoAccept: [] });
} else {
  const manifest = JSON.parse(readFileSync(resolve(corpusDir, 'manifest.json'), 'utf8')) as { samples: { file: string; expected_verses: Ref[]; also_accept?: Ref[][] }[] };
  for (const s of manifest.samples) clips.push({ file: s.file, path: resolve(corpusDir, s.file), expected: s.expected_verses, alsoAccept: s.also_accept ?? [] });
}

const ort = createRequire(resolve(frontend, 'package.json'))('onnxruntime-node');
const session = await ZipformerSession.create({ ort,
  model: new Uint8Array(readFileSync(resolve(frontend, 'public/models/zipformer_a0w_ep1_a05.int8.onnx'))),
  corpus: JSON.parse(readFileSync(resolve(frontend, 'public/zipformer_quran.json'), 'utf8')),
  quran: JSON.parse(readFileSync(resolve(frontend, 'public/quran.json'), 'utf8')),
});

const CHUNK = 2400; // 150 ms @ 16 kHz
const key = (r: Ref) => `${r.surah}:${r.ayah}`;
type Verdict = Record<string, unknown> & { state?: string; margin?: number; vowelErrors?: number; vowelMargin?: number; word?: string; wordIndex?: number };
interface ClipResult {
  file: string; durationSec: number; elapsedMs: number; refs: string[]; expected: string[]; expectedFound: boolean | null;
  flags: { atSec: number; message: WorkerOutbound }[]; verdicts: Verdict[]; verdictSnapshots: number;
}
const results: ClipResult[] = [];
const t0 = Date.now();

for (const clip of clips) {
  const buf = execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', clip.path, '-f', 'f32le', '-ar', '16000', '-ac', '1', 'pipe:1'], { maxBuffer: 200 * 1024 * 1024 });
  const pcm = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const started = Date.now();
  session.reset(); session.setMode('correction');
  const messages: { atSec: number; m: WorkerOutbound }[] = [];
  let last: Verdict[] = []; let lastJson = ''; let snapshots = 0;
  const snap = (atSec: number) => {
    const v = session.verdicts() as Verdict[];
    if (!v.length) return;
    const j = JSON.stringify(v);
    if (j === lastJson) return;
    lastJson = j; last = v; snapshots++;
    if (verbose) console.log(`  [${clip.file} @${atSec.toFixed(2)}s] verdicts:`, JSON.stringify(v));
  };
  for (let i = 0; i < pcm.length; i += CHUNK) {
    const atSec = (i + CHUNK) / 16000;
    for (const m of await session.feed(pcm.subarray(i, i + CHUNK))) messages.push({ atSec, m });
    snap(atSec);
  }
  for (const m of await session.stop()) messages.push({ atSec: pcm.length / 16000, m });
  snap(pcm.length / 16000);
  const refs = [...new Set(messages.filter(x => x.m.type === 'verse_match').map(x => key(x.m as unknown as Ref)))];
  const flags = messages.filter(x => x.m.type === 'correction' && (x.m as any).state?.phase !== 'idle').map(x => ({ atSec: x.atSec, message: x.m }));
  const expectedSets = [clip.expected, ...clip.alsoAccept].filter(s => s.length);
  const expectedFound = expectedSets.length ? expectedSets.some(s => s.every(r => refs.includes(key(r)))) : null;
  results.push({ file: clip.file, durationSec: pcm.length / 16000, elapsedMs: Date.now() - started, refs, expected: clip.expected.map(key), expectedFound, flags, verdicts: last, verdictSnapshots: snapshots });
  const flagStr = flags.map(f => { const is = (f.message as any).state?.issue; return `${is?.kind ?? '?'}@w${is?.wordIndex ?? '?'} ${f.atSec.toFixed(2)}s`; }).join(', ');
  console.log(`${clip.file.padEnd(32)} refs=[${refs.join(',')}] expected=${expectedFound === null ? 'n/a' : expectedFound ? 'OK' : 'MISS'} flags=${flags.length}${flagStr ? ' (' + flagStr + ')' : ''} words=${last.length}`);
  if (verbose && !flags.length) console.log('  final verdicts:', JSON.stringify(last));
}

// ---- summary ----
const allVerdicts = results.flatMap(r => r.verdicts.map(v => ({ file: r.file, v })));
const okMargins = allVerdicts.filter(x => x.v.state === 'ok' && typeof x.v.margin === 'number').map(x => x.v.margin as number);
const hist: Record<string, number> = {};
for (const m of okMargins) { const b = (Math.floor(m * 10) / 10).toFixed(1); hist[b] = (hist[b] ?? 0) + 1; }
const flagged = results.flatMap(r => r.flags.map(f => { const is = (f.message as any).state?.issue; return { file: r.file, word: is?.word, wordIndex: is?.wordIndex, kind: is?.kind, atSec: f.atSec, ref: is ? `${is.surah}:${is.ayah}` : undefined }; }));
const hasVowel = allVerdicts.some(x => 'vowelErrors' in x.v);
const vowel = hasVowel ? (() => {
  const hit = allVerdicts.filter(x => (x.v.state === 'ok' || x.v.state === 'unsure') && (x.v.vowelErrors as number) > 0);
  const perClip: Record<string, { count: number; vowelMargins: number[]; words: string[] }> = {};
  for (const { file, v } of hit) {
    const e = (perClip[file] ??= { count: 0, vowelMargins: [], words: [] });
    e.count++; e.words.push(`${v.word}#${v.wordIndex}`);
    if (typeof v.vowelMargin === 'number') e.vowelMargins.push(+v.vowelMargin.toFixed(3));
  }
  const vhist: Record<string, number> = {};
  for (const m of hit.map(x => x.v.vowelMargin).filter((m): m is number => typeof m === 'number')) { const b = (Math.floor(m * 10) / 10).toFixed(1); vhist[b] = (vhist[b] ?? 0) + 1; }
  return { okOrUnsureWithVowelErrors: hit.length, totalOkOrUnsure: allVerdicts.filter(x => x.v.state === 'ok' || x.v.state === 'unsure').length, vowelMarginHistogram: vhist, perClip };
})() : null;
const summary = {
  clips: results.length,
  withExpectations: results.filter(r => r.expectedFound !== null).length,
  expectedFound: results.filter(r => r.expectedFound === true).length,
  expectedMissed: results.filter(r => r.expectedFound === false).map(r => ({ file: r.file, expected: r.expected, refs: r.refs })),
  totalFalseFlags: flagged.length,
  clipsFlagged: new Set(flagged.map(f => f.file)).size,
  flagged,
  okWords: okMargins.length,
  okMarginHistogram: Object.fromEntries(Object.entries(hist).sort(([a], [b]) => +a - +b)),
  stateCounts: allVerdicts.reduce<Record<string, number>>((acc, x) => { const s = String(x.v.state); acc[s] = (acc[s] ?? 0) + 1; return acc; }, {}),
  verdictFields: [...new Set(allVerdicts.flatMap(x => Object.keys(x.v)))],
  vowel,
  totalAudioSec: +results.reduce((a, r) => a + r.durationSec, 0).toFixed(1),
  wallMs: Date.now() - t0,
};
console.log('\n=== SUMMARY ===\n' + JSON.stringify(summary, null, 2));
if (jsonOut) { writeFileSync(jsonOut, JSON.stringify({ summary, results }, null, 2)); console.log(`wrote ${jsonOut}`); }
// No process.exit(): onnxruntime-node aborts (mutex lock failed) when torn down mid-exit; let the loop drain.
process.exitCode = 0;
