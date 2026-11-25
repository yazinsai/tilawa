// Run after `npm ci && npm run build` in packages/tilawa-core. No runtime or audio required.
import { readFile, writeFile } from 'node:fs/promises';
import { Fbank } from '../../tilawa-core/dist/runtime/fbank.js';
import { Resampler } from '../../tilawa-core/dist/runtime/audio.js';
import { alignSubstring } from '../../tilawa-core/dist/engine/fuzzy.js';

const fixture = name => new URL(`../Tests/TilawaTests/Fixtures/${name}.json`, import.meta.url);
const save = (name, data) => writeFile(fixture(name), JSON.stringify(data) + '\n');
const samples = Float32Array.from({ length: 4800 }, (_, i) => 0.25 * Math.sin(2 * Math.PI * 440 * i / 16000) + 0.1 * Math.cos(2 * Math.PI * 123 * i / 16000));
const bank = new Fbank();
const frames = [...bank.push(samples.subarray(0, 317)), ...bank.push(samples.subarray(317, 2057)), ...bank.push(samples.subarray(2057))];
await save('fbank', { samples: Array.from(samples), frames: frames.map(f => Array.from(f)) });
const resampler = new Resampler(44100, 16000);
await save('resampler', { samples: Array.from(samples), output: [...resampler.push(samples.subarray(0, 317)), ...resampler.push(samples.subarray(317, 2057)), ...resampler.push(samples.subarray(2057)), ...resampler.flush()] });
await save('alignment', [
  ['بِسمِللَااه', 'بِسمِللَااه'], ['بِسمِللَااه', 'بِسمللَاه'],
  ['قُلهُوَللَاهُءَحَد', 'قُلهُوَللَاهُءَحَد'],
  ['ءَلحَمدُلِللَااهِرَببِ', 'لحَمدُلِللَاهِرَبب'],
  ['هحعءصسضدطتقكذزثس', 'ححءءسسددتتككززسس'],
].map(([reference, observed]) => {
  const result = alignSubstring(reference, 0, reference.length, observed, i => i === 0);
  return { reference, observed, start: result.start, end: result.end, cost: result.cost, errorRate: result.per };
}));
const reference = JSON.parse(await readFile(new URL('../../tilawa-core/assets/ref-v2.json', import.meta.url), 'utf8'));
await save('reference', { ayahs: reference.ayahs.filter(ayah => ayah.s === 36 && ayah.a <= 12) });
