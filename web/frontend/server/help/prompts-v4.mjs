// v4: correct recitation only. Volunteers are never asked to alter the Quran;
// accidental slips are reported by the reciter, never requested.
import { mulberry32 } from './prompts-v1.mjs';

export const PASSAGE_WEIGHTS = { single: 60, run: 40 };
const FAMILIAR = new Set([1, 18, 36, 55, 56, 67, ...Array.from({ length: 36 }, (_, i) => 78 + i)]);
const MAX_RUN_WORDS = 40;

const pick = (rng, xs) => xs[Math.floor(rng() * xs.length)];
const refOf = v => `${v.surah}:${v.ayah}`;
const pools = new WeakMap();

function poolsFor(quran) {
  if (pools.has(quran)) return pools.get(quran);
  const single = quran.list.filter(v => v.words.length >= 4 && v.words.length <= 22);
  // Consecutive ayahs exercise verse-boundary tracking, the weakest slice.
  const runs = [];
  quran.list.forEach((v, i) => {
    for (const len of [2, 3]) {
      const run = quran.list.slice(i, i + len);
      if (run.length !== len || run.some(r => r.surah !== v.surah)) continue;
      const words = run.reduce((a, r) => a + r.words.length, 0);
      if (words >= 6 && words <= MAX_RUN_WORDS) runs.push(run);
    }
  });
  const split = xs => ({ all: xs, familiar: xs.filter(x => FAMILIAR.has((x[0] ?? x).surah)) });
  const p = { single: split(single), run: split(runs) };
  pools.set(quran, p);
  return p;
}

const display = v => ({ ref: refOf(v), surahName: v.surahName, surahNameAr: v.surahNameAr, words: v.words.map((text, i) => ({ n: i + 1, text })) });

export function generatePrompt(quran, seed) {
  const rng = mulberry32(seed);
  const shape = rng() * (PASSAGE_WEIGHTS.single + PASSAGE_WEIGHTS.run) < PASSAGE_WEIGHTS.single ? 'single' : 'run';
  const pool = poolsFor(quran)[shape];
  const chosen = pick(rng, rng() < 0.5 && pool.familiar.length ? pool.familiar : pool.all);
  const verses = shape === 'single' ? [chosen] : chosen;
  const first = verses[0];
  const last = verses.at(-1);
  const ref = verses.length === 1 ? refOf(first) : `${refOf(first)}-${refOf(last)}`;
  return {
    v: 4,
    seed,
    label: { kind: 'none', ref, ayahs: verses.map(refOf), elicitation: 'normal_recitation', passage: shape, mistakes: [] },
    display: verses.map(display),
    instruction: verses.length === 1
      ? { en: 'Recite this ayah the way you normally would.', ar: 'اقرأ هذه الآية كما تقرؤها عادةً.' }
      : { en: `Recite these ${verses.length} ayahs in one go, the way you normally would.`, ar: `اقرأ ${verses.length === 2 ? 'هاتين الآيتين' : 'هذه الآيات الثلاث'} متتابعةً كما تقرؤها عادةً.` },
  };
}
