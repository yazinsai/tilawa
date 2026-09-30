import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { generatePrompt, KIND_WEIGHTS, loadQuran } from '../prompts.mjs';

const quran = loadQuran(JSON.parse(readFileSync(new URL('../../lab/data/quran.json', import.meta.url), 'utf8')));
const SEEDS = Array.from({ length: 3000 }, (_, i) => i * 7919 + 13);

test('prompts are deterministic per seed', () => {
  for (const seed of SEEDS.slice(0, 50)) assert.deepEqual(generatePrompt(quran, seed), generatePrompt(quran, seed));
});

test('every kind is generated, with no silent fallback to none', () => {
  for (const kind of Object.keys(KIND_WEIGHTS)) {
    for (const seed of SEEDS.slice(0, 200)) assert.equal(generatePrompt(quran, seed, kind).label.kind, kind);
  }
});

test('labels point at real words and match the display', () => {
  for (const seed of SEEDS) {
    const p = generatePrompt(quran, seed);
    const { kind, mistakes, ref } = p.label;
    assert.ok(p.instruction.en && p.instruction.ar);
    if (kind === 'none') {
      assert.deepEqual(mistakes, []);
      continue;
    }
    assert.equal(mistakes.length, 1);
    const m = mistakes[0];
    if (kind === 'skip_ayah') {
      const [a, , c] = p.display.map(d => d.ref);
      assert.equal(ref, `${a}-${c}`);
      assert.equal(p.display[1].mark, 'skip_ayah');
      assert.equal(m.ayah, p.display[1].ref);
      const [s1, a1] = a.split(':').map(Number);
      const [s3, a3] = c.split(':').map(Number);
      assert.equal(s1, s3);
      assert.equal(a3 - a1, 2);
      continue;
    }
    const verse = quran.byRef.get(m.ayah);
    assert.ok(m.word >= 1 && m.word <= verse.clean.length, `${seed} word index`);
    assert.equal(m.expected, verse.clean[m.word - 1]);
    const marked = p.display[0].words.flatMap((w, i) => (w.mark ? [i + 1] : []));
    assert.deepEqual(marked, [m.word], `${seed} highlighted word`);
    if (kind === 'substitution') assert.notEqual(m.said, m.expected);
    if (kind === 'vowel') {
      assert.notEqual(m.said_uthmani, verse.words[m.word - 1]);
      assert.notEqual(m.from, m.to);
    }
    if (kind === 'vowel') {
      const orig = [...verse.words[m.word - 1]];
      const said = [...m.said_uthmani];
      const at = orig.findIndex((c, i) => c !== said[i]);
      assert.equal(orig.filter((c, i) => c !== said[i]).length, 1, `${seed} exactly one vowel changed`);
      assert.doesNotMatch(said.slice(at).join(''), /^[\u064E-\u0650](\u0670|[اى](?![\u064B-\u0652]))/u, `${seed} vowel before madd`);
    }
    if (kind === 'tajweed') {
      assert.ok(['madd_short', 'no_ghunna'].includes(m.rule));
      const w = verse.words[m.word - 1];
      if (m.rule === 'madd_short') assert.match(w.slice(w.indexOf('\u0653') + 1), /[ءأإئؤ\u0651\u0652]/, `${seed} munfasil`);
    }
  }
});

test('mistake mix roughly follows the weights', () => {
  const counts = {};
  for (const seed of SEEDS) {
    const k = generatePrompt(quran, seed).label.kind;
    counts[k] = (counts[k] ?? 0) + 1;
  }
  const total = Object.values(KIND_WEIGHTS).reduce((a, b) => a + b, 0);
  for (const [k, w] of Object.entries(KIND_WEIGHTS)) {
    assert.ok(Math.abs(counts[k] / SEEDS.length - w / total) < 0.03, `${k}: ${counts[k]}`);
  }
});
