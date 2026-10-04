/** Local model/recording regression; no uploads. Checks false flags, not recall.
 * Run from web/frontend: node --import tsx test/correction-audio.ts */
import { readFileSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ZipformerSession, type WorkerOutbound } from '@tilawa/core';
const frontend = fileURLToPath(new URL('..', import.meta.url));
const repo = resolve(frontend, '../..');
const root = process.env.TILAWA_AUDIO_ROOT ?? [repo, resolve(repo, '../..')].find(p => existsSync(resolve(p, 'lab/benchmark/test_corpus/001002.mp3')))!;
const ort = createRequire(resolve(frontend, 'package.json'))('onnxruntime-node');
const session = await ZipformerSession.create({ ort,
  model: new Uint8Array(readFileSync(resolve(frontend, 'public/models/zipformer_a0w_ep1_a05.int8.onnx'))),
  corpus: JSON.parse(readFileSync(resolve(frontend, 'public/zipformer_quran.json'), 'utf8')),
  quran: JSON.parse(readFileSync(resolve(frontend, 'public/quran.json'), 'utf8')),
  // Structural rules are on by default; TILAWA_STRUCTURAL=0 turns them off, =pause runs them live.
  ...(process.env.TILAWA_STRUCTURAL === '0' ? { structural: false as const }
    : process.env.TILAWA_STRUCTURAL === 'pause' ? { structural: { ayahOrder: true, similarVerse: true, timing: 'pause' as const } } : {}),
});
const results: unknown[] = [];
for (const [file, expected] of [['001002.mp3', '1:2'], ['112001.mp3', '112:1'], ['ikhlas_2_3.m4a', '112:3'], ['002255.mp3', '2:255']] as const) {
  const buf = execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', resolve(root, 'lab/benchmark/test_corpus', file), '-f', 'f32le', '-ar', '16000', '-ac', '1', 'pipe:1'], { maxBuffer: 50 * 1024 * 1024 });
  const pcm = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  for (const mode of ['tracking', 'correction'] as const) {
    session.reset(); session.setMode(mode);
    const messages: WorkerOutbound[] = [];
    for (let i = 0; i < pcm.length; i += 7680) messages.push(...await session.feed(pcm.subarray(i, i + 7680)));
    messages.push(...await session.stop());
    const refs = messages.filter(m => m.type === 'verse_match').map(m => `${m.surah}:${m.ayah}`);
    const flags = messages.filter(m => m.type === 'correction');
    results.push({ file, mode, refs, flags });
    if (!refs.includes(expected) || flags.length) throw new Error(JSON.stringify(results, null, 2));
  }
}
session.reset(); session.setMode('correction');
for (let i = 0; i < 20; i++) await session.feed(new Float32Array(7680));
if (session.correction.state.phase !== 'idle') throw new Error('Silence flagged');
console.log(JSON.stringify({ results, silence: 'no flags' }, null, 2));
