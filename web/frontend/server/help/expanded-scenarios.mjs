import { createHash } from 'node:crypto';
import { SCENARIOS } from './scenarios.mjs';

const LETTER = /[\u0621-\u063a\u0641-\u064a\u0671-\u06d3]/u;
const VOWELS = { '\u064e': 'fatha', '\u064f': 'damma', '\u0650': 'kasra' };
const SHORT_WORDS = new Set(['من', 'في', 'ما', 'لا', 'لم', 'لن', 'ان', 'الا', 'قد', 'مع', 'عن', 'ثم', 'او', 'بل', 'كل', 'هو', 'هي', 'هم', 'وما', 'ولا', 'فلا', 'فلم']);
const normalize = word => word.replace(/[\u0640\u064b-\u065f\u0670\u06d6-\u06ed]/gu, '').replace(/[أإآٱ]/gu, 'ا');
const refOf = v => `${v.surah}:${v.ayah}`;
const cache = new WeakMap();

export function letters(word) {
  const result = [];
  for (const c of word) {
    if (LETTER.test(c)) result.push({ base: c, marks: [] });
    else if (result.length) result.at(-1).marks.push(c);
  }
  return result;
}

function canonical(word) {
  // Keep shadda/sukun/madd and vowel identities; remove annotation-only signs.
  return letters(word).map(g => ({ base: g.base === 'ٱ' ? 'ا' : g.base,
    marks: g.marks.filter(m => !/[\u0640\u06d6-\u06ed]/u.test(m)).sort() }));
}

function shortVowelSite(gs, i) {
  if (i === gs.length - 1 || gs[i].marks.includes('\u0670') || gs[i].marks.includes('\u0653')) return false;
  const next = gs[i + 1];
  // No changing long vowels or diphthongs into invented spellings.
  if (/[اىوي]/u.test(next.base) && !next.marks.some(m => m in VOWELS || m === '\u0651')) return false;
  return gs[i].marks.filter(m => m in VOWELS).length === 1;
}

function fingerprint(s) {
  const mutation = s.kind === 'substitution' ? (s.source ? ['word', s.altKey] : ['letter', s.letterPosition, s.toLetter])
    : s.kind === 'vowel' ? [s.letterPosition, s.from, s.to]
    : s.kind === 'tajweed' ? [s.rule, s.letterPosition] : [];
  return JSON.stringify([s.kind, s.ref, s.word, s.skip, s.end, mutation]);
}

export function buildScenarioBank(quran) {
  if (cache.has(quran)) return cache.get(quran);
  const scenarios = [];
  const seen = new Set();
  const add = s => {
    const key = fingerprint(s);
    if (seen.has(key)) return;
    seen.add(key);
    scenarios.push(s);
  };
  const pool = quran.list.filter(v => v.words.length >= 2 && v.words.length <= 22 && v.words.length === v.clean.length);
  const sites = [];
  const vocabulary = new Map();
  const passageSignatures = new Map();
  for (const v of pool) {
    const ref = refOf(v);
    const keys = v.words.map(normalize);
    for (let j = 0; j < v.words.length; j++) {
      // Do not inflate the bank with the same basmala in every opening ayah.
      if (v.ayah === 1 && v.surah !== 1 && j < 4 && keys.slice(0, 4).join(' ') === 'بسم الله الرحمن الرحيم') continue;
      const word = j + 1;
      const gs = letters(v.words[j]);
      const canonicalGs = canonical(v.words[j]);
      const site = { ref, word, gs, canonicalGs, verse: v };
      sites.push(site);
      for (let i = 0; i < gs.length; i++) {
        const g = gs[i];
        const common = { ref, word, letter: g.base, letterPosition: i + 1 };
        // Expand the observed qaf/kaf mechanism across real lexical contexts.
        if (g.base === 'ق') add({ ...common, id: `qaf-kaf:${ref}:${word}:${i + 1}`, kind: 'substitution', mechanism: 'letter_confusion', fromLetter: 'ق', toLetter: 'ك' });
        if ('صضطظ'.includes(g.base)) add({ ...common, id: `light:${ref}:${word}:${i + 1}`, kind: 'tajweed', mechanism: 'emphasis_loss', rule: 'light_emphatic' });
        if ('نم'.includes(g.base) && g.marks.includes('\u0651')) add({ ...common, id: `ghunna:${ref}:${word}:${i + 1}`, kind: 'tajweed', mechanism: 'ghunna_duration', rule: 'no_ghunna' });
        const solarArticle = i >= 2 && gs[i - 1].base === 'ل' && /[اٱ]/u.test(gs[i - 2].base);
        if (i > 0 && g.marks.includes('\u0651') && !solarArticle) add({ ...common, id: `shadda:${ref}:${word}:${i + 1}`, kind: 'tajweed', mechanism: 'gemination_loss', rule: 'no_shadda' });
        if (g.marks.includes('\u0653') && /[اوى]/u.test(g.base) && gs[i + 1]?.marks.includes('\u0651') && !(i < 3 && gs.slice(0, i).some(g => g.base === 'ء'))) add({ ...common, id: `madd:${ref}:${word}:${i + 1}`, kind: 'tajweed', mechanism: 'madd_duration', rule: 'madd_short' });
        if (shortVowelSite(canonicalGs, i)) {
          const vowel = canonicalGs[i].marks.find(m => m in VOWELS);
          const signature = canonicalGs.map((g, index) => [g.base, g.marks.map(m => index === i && m in VOWELS ? '*' : m)]);
          const key = JSON.stringify(signature);
          const entry = { ...common, vowel, source: ref, sourceWord: word };
          const variants = vocabulary.get(key) ?? new Map();
          if (!variants.has(vowel)) variants.set(vowel, entry);
          vocabulary.set(key, variants);
          site.vowelSites ??= [];
          site.vowelSites.push({ i, vowel, key });
        }
      }
      if (j > 0 && j < keys.length - 1) {
        const bare = keys[j].replace(/^و(?=.{2})/u, '');
        const repeated = keys.filter(k => k.replace(/^و(?=.{2})/u, '') === bare).length > 1;
        if (SHORT_WORDS.has(keys[j]) || repeated) add({ id: `omit:${ref}:${word}`, kind: 'skip_word', mechanism: repeated ? 'repeated_phrase_omission' : 'short_word_omission', ref, word });
        if (keys[j] !== keys[j - 1] && keys[j] !== keys[j + 1]) add({ id: `hesitate:${ref}:${word}`, kind: 'repeat', mechanism: 'hesitation_repeat', ref, word });
      }
      // Near-parallel passages must share at least four unchanged words.
      if (keys.length >= 5) {
        const signature = JSON.stringify(keys.map((k, at) => at === j ? '*' : k));
        const group = passageSignatures.get(signature) ?? [];
        group.push({ ref, word, key: keys[j], verse: v });
        passageSignatures.set(signature, group);
      }
    }
  }
  for (const site of sites) {
    for (const { i, vowel, key } of site.vowelSites ?? []) {
      for (const [other, source] of vocabulary.get(key)) {
        if (other === vowel) continue;
        add({ id: `vowel:${site.ref}:${site.word}:${i + 1}:${VOWELS[other]}`, kind: 'vowel', mechanism: 'attested_vowel_confusion', ref: site.ref, word: site.word, letter: site.gs[i].base, letterPosition: i + 1, from: VOWELS[vowel], to: VOWELS[other], vowelSource: `${source.source}:${source.sourceWord}` });
      }
    }
  }
  for (const group of passageSignatures.values()) {
    for (const target of group) {
      const alternatives = new Map();
      for (const source of group) {
        if (source.ref === target.ref || source.key === target.key) continue;
        // Ignore orthographic/long-vowel-only differences, which can reflect
        // transmitted readings rather than the intended lexical confusion.
        if (source.key.replace(/[اوىي]/gu, '') === target.key.replace(/[اوىي]/gu, '')) continue;
        if (!alternatives.has(source.key)) alternatives.set(source.key, source);
      }
      for (const source of alternatives.values()) add({ id: `parallel:${target.ref}:${target.word}:${source.ref}:${source.word}`, kind: 'substitution', mechanism: 'similar_passage', ref: target.ref, word: target.word, source: source.ref, sourceWord: source.word, altKey: source.key });
    }
  }
  for (const first of quran.list) {
    const middle = quran.byRef.get(`${first.surah}:${first.ayah + 1}`);
    const end = quran.byRef.get(`${first.surah}:${first.ayah + 2}`);
    const run = [first, middle, end];
    if (!middle || !end || run.some(v => v.words.length < 2 || v.words.length > 14 || v.words.length !== v.clean.length) || run.reduce((n, v) => n + v.words.length, 0) > 32) continue;
    const words = run.map(v => v.words.map(normalize));
    let mechanism;
    for (let a = 0; a < 3; a++) for (let b = a + 1; b < 3; b++) {
      if (words[a].at(-1) === words[b].at(-1)) mechanism ??= 'similar_ending_omission';
      if (words[a].slice(0, 2).join(' ') === words[b].slice(0, 2).join(' ')) mechanism = 'parallel_verse_omission';
    }
    if (mechanism) add({ id: `skip:${refOf(middle)}`, kind: 'skip_ayah', mechanism, ref: refOf(first), skip: refOf(middle), end: refOf(end) });
  }
  // Retain the original pedagogical examples, including explicitly continuous
  // reading for the person-ending vowel in an'amta.
  for (const s of SCENARIOS) {
    if (!quran.byRef.has(s.ref) || (s.source && !quran.byRef.has(s.source)) || (s.skip && (!quran.byRef.has(s.skip) || !quran.byRef.has(s.end)))) continue;
    const site = s.word ? letters(quran.byRef.get(s.ref).words[s.word - 1]) : [];
    const letter = s.letter ?? s.fromLetter ?? (s.rule === 'madd_short' ? site.find(g => g.marks.includes('\u0653'))?.base : undefined);
    const letterPosition = letter ? site.findIndex(g => g.base === letter) + 1 : undefined;
    add({ ...s, ...(letterPosition ? { letterPosition, letter } : {}), ...(s.source ? { altKey: normalize(quran.byRef.get(s.source).words[s.sourceWord - 1]) } : {}) });
  }
  // Group identical passages, near-parallel substitutions and the whole
  // context of skipped-ayah prompts. These groups must not cross data splits.
  const parent = new Map(quran.list.map(v => [refOf(v), refOf(v)]));
  const find = ref => {
    let root = ref;
    while (parent.get(root) !== root) root = parent.get(root);
    while (ref !== root) { const next = parent.get(ref); parent.set(ref, root); ref = next; }
    return root;
  };
  const union = (a, b) => { const x = find(a), y = find(b); if (x !== y) parent.set(y, x); };
  const identical = new Map();
  for (const v of quran.list) {
    const key = v.words.map(normalize).join(' ');
    if (identical.has(key)) union(refOf(v), identical.get(key));
    else identical.set(key, refOf(v));
  }
  for (const s of scenarios) {
    if (s.source) union(s.ref, s.source);
    if (s.skip) { union(s.ref, s.skip); union(s.ref, s.end); }
  }
  const groups = new Map();
  for (const ref of parent.keys()) { const root = find(ref); const refs = groups.get(root) ?? []; refs.push(ref); groups.set(root, refs); }
  const families = new Map();
  const familySizes = new Map();
  for (const refs of groups.values()) {
    const id = `pf3-${createHash('sha256').update(refs.sort().join('|')).digest('hex').slice(0, 16)}`;
    familySizes.set(id, refs.length);
    for (const ref of refs) families.set(ref, id);
  }
  const bank = { scenarios, normal: pool, families, familySizes };
  cache.set(quran, bank);
  return bank;
}

export function diversitySummary(quran) {
  const bank = buildScenarioBank(quran);
  const lexicalTemplates = new Set(bank.scenarios.map(s => {
    const target = quran.byRef.get(s.ref).words.map(normalize).join(' ');
    const span = s.skip ? [s.skip, s.end].map(ref => quran.byRef.get(ref).words.map(normalize).join(' ')) : [];
    return JSON.stringify([s.kind, target, s.word, span, s.rule, s.letterPosition, s.from, s.to, s.toLetter, s.altKey]);
  }));
  const kinds = {};
  for (const s of bank.scenarios) {
    const group = kinds[s.kind] ??= { scenarios: 0, verses: new Set(), surahs: new Set(), mechanisms: {} };
    group.scenarios++;
    group.verses.add(s.ref);
    group.surahs.add(Number(s.ref.split(':')[0]));
    group.mechanisms[s.mechanism] = (group.mechanisms[s.mechanism] ?? 0) + 1;
  }
  return { version: 3, uniqueLexicalTemplates: lexicalTemplates.size, passageFamilies: bank.familySizes.size, largestPassageFamily: Math.max(...bank.familySizes.values()), mistakeScenarios: bank.scenarios.length, normalPassages: bank.normal.length,
    kinds: Object.fromEntries(Object.entries(kinds).map(([k, v]) => [k, { ...v, verses: v.verses.size, surahs: v.surahs.size }])),
    uniqueTargetPassages: new Set(bank.scenarios.map(s => s.ref)).size,
    uniqueTargetSurahs: new Set(bank.scenarios.map(s => Number(s.ref.split(':')[0]))).size };
}
