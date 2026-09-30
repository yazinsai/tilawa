import { generatePrompt as generateLegacyPrompt, loadQuran, mulberry32, KIND_WEIGHTS } from './prompts-v1.mjs';
import { SCENARIOS } from './scenarios.mjs';
export { loadQuran, mulberry32, KIND_WEIGHTS, SCENARIOS };

const VOWELS = { fatha: '\u064e', damma: '\u064f', kasra: '\u0650' };
const VOWEL_AR = { fatha: 'فتحة', damma: 'ضمة', kasra: 'كسرة' };
const LETTER = /[\u0621-\u064a\u0671-\u06d3]/u;
const pick = (rng, xs) => xs[Math.floor(rng() * xs.length)];
const refOf = v => `${v.surah}:${v.ayah}`;
const bankCache = new WeakMap();

// Build once per corpus. Missing verses in a partial corpus are allowed; a
// mismatched existing site is a data error, never a silent random replacement.
function scenarioBank(quran) {
  if (bankCache.has(quran)) return bankCache.get(quran);
  const bank = {};
  for (const scenario of SCENARIOS) {
    const v = quran.byRef.get(scenario.ref);
    if (!v || (scenario.source && !quran.byRef.has(scenario.source)) || (scenario.skip && (!quran.byRef.has(scenario.skip) || !quran.byRef.has(scenario.end)))) continue;
    if (v.words.length !== v.clean.length || (scenario.word && !v.words[scenario.word - 1])) throw new Error(`Invalid scenario site: ${scenario.id}`);
    (bank[scenario.kind] ??= []).push(scenario);
  }
  bankCache.set(quran, bank);
  return bank;
}

function display(v, word, mark) {
  return { ref: refOf(v), surahName: v.surahName, surahNameAr: v.surahNameAr,
    words: v.words.flatMap((text, i) => {
      const w = { n: i + 1, text, ...(i + 1 === word ? mark : {}) };
      return w.mark === 'repeat' ? [w, { ...w }] : [w];
    }) };
}

function modifyLetter(word, letter, mutation) {
  const chars = [...word];
  const at = chars.indexOf(letter);
  if (at < 0) throw new Error(`Missing ${letter} in ${word}`);
  let end = at + 1;
  while (end < chars.length && !LETTER.test(chars[end])) end++;
  const marks = chars.slice(at + 1, end);
  mutation(chars, at, marks);
  return chars.join('');
}

export function generatePrompt(quran, seed, kind, version = 2) {
  if (version === 1) return generateLegacyPrompt(quran, seed, kind);
  if (version !== 2) throw new Error('Unsupported prompt version');
  const rng = mulberry32(seed);
  if (kind === undefined) {
    let roll = rng() * Object.values(KIND_WEIGHTS).reduce((a, b) => a + b, 0);
    kind = Object.entries(KIND_WEIGHTS).find(([, weight]) => (roll -= weight) < 0)?.[0] ?? 'none';
  }
  if (!(kind in KIND_WEIGHTS)) throw new Error('Unknown prompt kind');
  if (kind === 'none') {
    const legacy = generateLegacyPrompt(quran, seed, 'none');
    return { ...legacy, v: 2, label: { ...legacy.label, elicitation: 'normal_recitation' } };
  }
  const candidates = scenarioBank(quran)[kind];
  if (!candidates?.length) throw new Error(`No curated scenarios for ${kind} in this corpus`);
  const s = pick(rng, candidates);
  const v = quran.byRef.get(s.ref);
  const label = { kind, ref: s.ref, elicitation: 'acted_scenario', scenario: s.id, mechanism: s.mechanism, mistakes: [] };
  let instruction;
  if (kind === 'skip_ayah') {
    const verses = [v, quran.byRef.get(s.skip), quran.byRef.get(s.end)];
    label.ref = `${s.ref}-${s.end}`;
    label.mistakes = [{ kind, ayah: s.skip }];
    instruction = { en: `Read these ayahs, skipping the crossed-out ayah ${s.skip}, as if you lost your place between similar lines.`, ar: `اقرأ هذه الآيات واترك الآية المشطوبة ${s.skip}، كأنك فقدت موضعك بين آيات متشابهة.` };
    return { v: 2, seed, label, instruction, display: verses.map((verse, i) => ({ ...display(verse), ...(i === 1 ? { mark: 'skip_ayah' } : {}) })) };
  }
  const w = v.words[s.word - 1];
  const m = { kind, ayah: s.ref, word: s.word, expected: v.clean[s.word - 1] };
  const mark = { mark: kind };
  if (kind === 'substitution') {
    let alt;
    if (s.source) {
      const source = quran.byRef.get(s.source);
      alt = source.words[s.sourceWord - 1];
      if (!alt || source.words.length !== source.clean.length) throw new Error(`Invalid source: ${s.id}`);
      // These curated pairs have matching neighbours, so retain contextual
      // Uthmani marks instead of indiscriminately deleting them.
      m.said = source.clean[s.sourceWord - 1];
      m.said_from = `${s.source}:${s.sourceWord}`;
      instruction = { en: 'Read exactly as shown. The highlighted word comes from a similar ayah, as if you mixed up the two passages.', ar: 'اقرأ كما هو مكتوب. الكلمة المظللة من آية مشابهة، كأنك خلطت بين الموضعين.' };
    } else {
      alt = modifyLetter(w, s.fromLetter, (chars, at) => { chars[at] = s.toLetter; });
      m.said = m.expected.replace(s.fromLetter, s.toLetter);
      m.rule = 'letter_confusion';
      m.from_letter = s.fromLetter;
      m.to_letter = s.toLetter;
      instruction = { en: `In the highlighted word, say «${s.toLetter}» instead of «${s.fromLetter}», exactly as shown. Keep the other letters and vowels unchanged.`, ar: `في الكلمة المظللة انطق «${s.toLetter}» بدل «${s.fromLetter}» كما هو مكتوب، وأبقِ بقية الحروف والحركات كما هي.` };
    }
    if (m.said === m.expected) throw new Error(`Unchanged substitution: ${s.id}`);
    m.said_uthmani = mark.text = alt;
  } else if (kind === 'vowel') {
    let letterIndex;
    const alt = modifyLetter(w, s.letter, (chars, at, marks) => {
      const offset = marks.indexOf(VOWELS[s.from]);
      if (offset < 0) throw new Error(`Missing vowel: ${s.id}`);
      letterIndex = chars.slice(0, at + 1).filter(c => LETTER.test(c)).length;
      chars[at + 1 + offset] = VOWELS[s.to];
    });
    Object.assign(m, { said_uthmani: alt, letter: letterIndex, from: s.from, to: s.to, ...(s.continue ? { continuous_reading: true } : {}) });
    mark.text = alt;
    instruction = { en: `Read as shown: on «${s.letter}» in the highlighted word, use ${s.to} instead of ${s.from}.${s.continue ? ' Continue directly into the next word without pausing, so the changed vowel is audible.' : ' Read the rest normally.'}`, ar: `اقرأ كما هو مكتوب: على «${s.letter}» في الكلمة المظللة ${VOWEL_AR[s.to]} بدل ${VOWEL_AR[s.from]}.${s.continue ? ' أكمل إلى الكلمة التالية دون وقف، حتى تُسمع الحركة المتغيّرة.' : ' اقرأ الباقي كالمعتاد.'}` };
  } else if (kind === 'skip_word') {
    instruction = { en: `Skip only the highlighted word «${w}», as if you missed it while reading. Read the rest normally.`, ar: `اترك الكلمة المظللة «${w}» فقط، كأنك تجاوزتها أثناء القراءة. اقرأ الباقي كالمعتاد.` };
  } else if (kind === 'repeat') {
    instruction = { en: 'Read as shown: say the highlighted word twice, with a brief hesitation before repeating it. Then continue normally.', ar: 'اقرأ كما هو مكتوب: قل الكلمة المظللة مرتين، مع تردّد بسيط قبل إعادتها، ثم أكمل كالمعتاد.' };
    // Keep the legacy class, but identify this as a fluency event rather than
    // asserting that every legitimate restart is a recitation violation.
    label.fluency_control = true;
  } else if (kind === 'tajweed') {
    m.rule = mark.rule = s.rule;
    if (s.rule === 'madd_short') {
      if (!w.includes('\u0653') || !w.includes('\u0651')) throw new Error(`Missing madd lazim: ${s.id}`);
      instruction = { en: 'In the highlighted word, shorten the internal alif madd to 2 counts instead of 6. Keep the doubled letters and the ending normal.', ar: 'في الكلمة المظللة اقصر مد الألف الداخلي إلى حركتين بدل ست حركات. أبقِ التشديد ونهاية الكلمة كالمعتاد.' };
    } else if (s.rule === 'no_ghunna') {
      modifyLetter(w, s.letter, (_, __, marks) => { if (!marks.includes('\u0651')) throw new Error(`Missing doubled letter: ${s.id}`); });
      instruction = { en: `Keep the doubled ${s.letter === 'م' ? 'meem' : 'noon'} in the highlighted word, but omit its sustained nasal hum (ghunna). Read everything else normally.`, ar: `أبقِ تشديد ${s.letter === 'م' ? 'الميم' : 'النون'} في الكلمة المظللة، لكن دون إطالة الغنة. اقرأ الباقي كالمعتاد.` };
    } else if (s.rule === 'light_emphatic') {
      if (!w.includes(s.letter)) throw new Error(`Missing emphatic letter: ${s.id}`);
      m.letter = s.letter;
      instruction = { en: 'In the highlighted word, read the sad (ص) too lightly, without its usual heavy resonance. Keep it a sad, not a seen (س), and read the rest normally.', ar: 'في الكلمة المظللة اقرأ الصاد بترقيق بدل التفخيم، مع إبقائها صادًا لا سينًا. اقرأ الباقي كالمعتاد.' };
    } else if (s.rule === 'no_shadda') {
      const alt = modifyLetter(w, s.letter, (chars, at, marks) => {
        const offset = marks.indexOf('\u0651');
        if (offset < 0) throw new Error(`Missing shadda: ${s.id}`);
        chars.splice(at + 1 + offset, 1);
      });
      m.said_uthmani = mark.text = alt;
      instruction = { en: 'Read as shown: pronounce the noon in the highlighted word once, without doubling it. Keep the other letters and vowels normal.', ar: 'اقرأ كما هو مكتوب: انطق النون في الكلمة المظللة مرة واحدة دون تشديد، وأبقِ بقية الحروف والحركات كالمعتاد.' };
    }
  }
  label.mistakes = [m];
  return { v: 2, seed, label, instruction, display: [display(v, s.word, mark)] };
}
