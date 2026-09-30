import { test } from 'vitest';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildScenarioBank, diversitySummary, generatePrompt, loadQuran, renderScenario } from '../server/help/prompts.mjs';
import { letters } from '../server/help/expanded-scenarios.mjs';

const quran = loadQuran(JSON.parse(readFileSync(new URL('../public/quran.json', import.meta.url), 'utf8')));
const bank = buildScenarioBank(quran);

test('the bank provides broad lexical, passage and class coverage', () => {
  const stats = diversitySummary(quran);
  assert.ok(stats.mistakeScenarios > 70000);
  assert.ok(stats.uniqueLexicalTemplates > 60000);
  assert.ok(stats.normalPassages > 5000);
  assert.equal(stats.uniqueTargetSurahs, 114);
  for (const [kind, minimum] of Object.entries({ substitution: 4000, vowel: 3000, tajweed: 15000, skip_word: 7000, repeat: 30000, skip_ayah: 150 })) {
    assert.ok(stats.kinds[kind].scenarios >= minimum, kind);
    assert.ok(stats.kinds[kind].surahs >= 50, kind);
  }
});

test('every expanded site produces a valid, unambiguous label and script', () => {
  const ids = new Set();
  for (const s of bank.scenarios) {
    assert.ok(!ids.has(s.id), s.id);
    ids.add(s.id);
    const p = renderScenario(quran, 123, s);
    assert.ok(p.instruction.en && p.instruction.ar, s.id);
    assert.equal(p.v, 3);
    assert.equal(p.label.scenario, s.id);
    assert.ok(p.label.passage_family);
    const m = p.label.mistakes[0];
    if (s.kind === 'skip_ayah') {
      const [first, middle, end] = p.display.map(d => d.ref.split(':').map(Number));
      assert.equal(first[0], middle[0]); assert.equal(first[0], end[0]);
      assert.equal(first[1] + 1, middle[1]); assert.equal(first[1] + 2, end[1]);
      assert.equal(p.display[1].mark, 'skip_ayah');
      continue;
    }
    const v = quran.byRef.get(s.ref);
    assert.equal(m.expected, v.clean[s.word - 1], s.id);
    const marked = p.display[0].words.filter(w => w.mark);
    assert.deepEqual(marked.map(w => w.n), s.kind === 'repeat' ? [s.word, s.word] : [s.word], s.id);
    if (m.said_uthmani) assert.equal(marked[0].text, m.said_uthmani, s.id);
    const original = letters(v.words[s.word - 1]);
    if (s.fromLetter) {
      const altered = letters(m.said_uthmani);
      assert.equal(altered.length, original.length);
      for (let i = 0; i < original.length; i++) {
        assert.equal(altered[i].base, i + 1 === s.letterPosition ? s.toLetter : original[i].base, s.id);
        assert.deepEqual(altered[i].marks, original[i].marks, s.id);
      }
      assert.notEqual(m.said, m.expected, s.id);
    }
    if (s.kind === 'vowel') {
      const before = [...v.words[s.word - 1]], after = [...m.said_uthmani];
      assert.equal(before.length, after.length);
      assert.equal(before.filter((c, i) => c !== after[i]).length, 1, s.id);
      if (!s.continue) assert.ok(s.letterPosition < original.length, s.id);
      if (s.mechanism === 'attested_vowel_confusion') assert.ok(m.vowel_source, s.id);
    }
    if (s.rule === 'no_shadda') {
      assert.equal([...v.words[s.word - 1]].length - [...m.said_uthmani].length, 1, s.id);
      assert.ok(original[s.letterPosition - 1].marks.includes('\u0651'), s.id);
      assert.ok(!letters(m.said_uthmani)[s.letterPosition - 1].marks.includes('\u0651'), s.id);
    }
    if (s.rule === 'madd_short') {
      const g = original[s.letterPosition - 1], next = original[s.letterPosition];
      assert.ok(g.marks.includes('\u0653'), s.id);
      assert.ok(next.marks.some(m => m === '\u0651' || m === '\u0652'), s.id);
    }
  }
});

test('balanced sampling retains rare mechanisms and distributes passages', () => {
  const counts = {};
  const seen = new Set();
  for (let i = 0; i < 4000; i++) {
    const p = generatePrompt(quran, i * 7919 + 13, 'tajweed');
    counts[p.label.mechanism] = (counts[p.label.mechanism] ?? 0) + 1;
    seen.add(p.label.ref);
  }
  for (const count of Object.values(counts)) assert.ok(Math.abs(count / 4000 - .25) < .04);
  assert.ok(seen.size > 1000);
  const vowels = new Set();
  for (let i = 0; i < 1000; i++) vowels.add(generatePrompt(quran, i * 7919 + 13, 'vowel').label.ref);
  assert.ok(vowels.size > 500, 'tiny anchor mechanisms cannot dominate vowel sampling');
});

test('identical and parallel passages and skip contexts stay in one family', () => {
  for (const s of bank.scenarios) {
    const family = bank.families.get(s.ref);
    for (const ref of [s.source, s.skip, s.end].filter(Boolean)) assert.equal(bank.families.get(ref), family, s.id);
  }
  assert.equal(bank.families.get('109:3'), bank.families.get('109:5'));
  assert.throws(() => generatePrompt(quran, 1, 'constructor'), /Unknown/);
});
