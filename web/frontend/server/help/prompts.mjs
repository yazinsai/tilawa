import { generatePrompt as generateLegacyPrompt, loadQuran, mulberry32, KIND_WEIGHTS } from './prompts-v1.mjs';
import { SCENARIOS } from './scenarios.mjs';
import { generatePrompt as generateV2Prompt } from './prompts-v2.mjs';
import { buildScenarioBank, diversitySummary } from './expanded-scenarios.mjs';
export { buildScenarioBank, diversitySummary };
export { loadQuran, mulberry32, KIND_WEIGHTS, SCENARIOS };

const VOWELS = { fatha: '\u064e', damma: '\u064f', kasra: '\u0650' };
const VOWEL_AR = { fatha: 'فتحة', damma: 'ضمة', kasra: 'كسرة' };
const LETTER = /[\u0621-\u063a\u0641-\u064a\u0671-\u06d3]/u;
const pick = (rng, xs) => xs[Math.floor(rng() * xs.length)];
const passageFamily = (quran, v) => buildScenarioBank(quran).families.get(refOf(v));
const lettersBefore = (word, position, letter) => [...word].filter(c => LETTER.test(c)).slice(0, (position ?? 1) - 1).filter(c => c === letter).length;
const refOf = v => `${v.surah}:${v.ayah}`;
const bankCache = new WeakMap();

// Build once per corpus. Missing verses in a partial corpus are allowed; a
// mismatched existing site is a data error, never a silent random replacement.
function scenarioBank(quran) {
  if (bankCache.has(quran)) return bankCache.get(quran);
  const bank = {};
  for (const scenario of buildScenarioBank(quran).scenarios) {
    const kind = bank[scenario.kind] ??= new Map();
    const bucket = scenario.kind === 'vowel' ? 'vowel_change' : scenario.mechanism;
    const mechanism = kind.get(bucket) ?? new Map();
    const surah = Number(scenario.ref.split(':')[0]);
    const verses = mechanism.get(surah) ?? new Map();
    const sites = verses.get(scenario.ref) ?? [];
    sites.push(scenario);
    verses.set(scenario.ref, sites);
    mechanism.set(surah, verses);
    kind.set(bucket, mechanism);
  }
  bankCache.set(quran, bank);
  return bank;
}

// Balance mechanisms, surahs and verses before sites: abundant word repeats or
// doubled noons must not crowd out parallel-passage errors and rare madd sites.
function pickScenario(rng, groups) {
  let selected = groups;
  while (selected instanceof Map) selected = pick(rng, [...selected.values()]);
  return pick(rng, selected);
}

function display(v, word, mark) {
  return { ref: refOf(v), surahName: v.surahName, surahNameAr: v.surahNameAr,
    words: v.words.flatMap((text, i) => {
      const w = { n: i + 1, text, ...(i + 1 === word ? mark : {}) };
      return w.mark === 'repeat' ? [w, { ...w }] : [w];
    }) };
}

function modifyLetter(word, letter, mutation, position) {
  const chars = [...word];
  const positions = chars.map((c, i) => LETTER.test(c) ? i : -1).filter(i => i >= 0);
  const at = position ? positions[position - 1] : chars.indexOf(letter);
  if (chars[at] !== letter) throw new Error(`Wrong letter position in ${word}`);
  if (at < 0) throw new Error(`Missing ${letter} in ${word}`);
  let end = at + 1;
  while (end < chars.length && !LETTER.test(chars[end])) end++;
  const marks = chars.slice(at + 1, end);
  mutation(chars, at, marks);
  return chars.join('');
}

export function generatePrompt(quran, seed, kind, version = 3) {
  if (version === 1) return generateLegacyPrompt(quran, seed, kind);
  if (version === 2) return generateV2Prompt(quran, seed, kind);
  if (version !== 3) throw new Error('Unsupported prompt version');
  const rng = mulberry32(seed);
  if (kind === undefined) {
    let roll = rng() * Object.values(KIND_WEIGHTS).reduce((a, b) => a + b, 0);
    kind = Object.entries(KIND_WEIGHTS).find(([, weight]) => (roll -= weight) < 0)?.[0] ?? 'none';
  }
  if (!Object.hasOwn(KIND_WEIGHTS, kind)) throw new Error('Unknown prompt kind');
  if (kind === 'none') {
    const v = pick(rng, buildScenarioBank(quran).normal);
    if (!v) throw new Error('No eligible normal passages');
    return { v: 3, seed, label: { kind, ref: refOf(v), elicitation: 'normal_recitation', passage_family: passageFamily(quran, v), mistakes: [] }, display: [display(v)], instruction: { en: 'Read it correctly, the way you normally would.', ar: 'اقرأ الآية قراءة صحيحة كما تقرؤها عادةً.' } };
  }
  const candidates = scenarioBank(quran)[kind];
  if (!candidates?.size) throw new Error(`No curated scenarios for ${kind} in this corpus`);
  const s = pickScenario(rng, candidates);
  return renderScenario(quran, seed, s);
}

export function renderScenario(quran, seed, s) {
  const kind = s.kind;
  const v = quran.byRef.get(s.ref);
  const label = { kind, ref: s.ref, elicitation: 'acted_scenario', scenario: s.id, mechanism: s.mechanism, passage_family: passageFamily(quran, v), mistakes: [] };
  let instruction;
  if (kind === 'skip_ayah') {
    const verses = [v, quran.byRef.get(s.skip), quran.byRef.get(s.end)];
    label.ref = `${s.ref}-${s.end}`;
    label.mistakes = [{ kind, ayah: s.skip }];
    instruction = { en: `Read these ayahs, skipping the crossed-out ayah ${s.skip}, as if you lost your place between similar lines.`, ar: `اقرأ هذه الآيات واترك الآية المشطوبة ${s.skip}، كأنك فقدت موضعك بين آيات متشابهة.` };
    return { v: 3, seed, label, instruction, display: verses.map((verse, i) => ({ ...display(verse), ...(i === 1 ? { mark: 'skip_ayah' } : {}) })) };
  }
  const w = v.words[s.word - 1];
  const m = { kind, ayah: s.ref, word: s.word, expected: v.clean[s.word - 1], ...(s.letterPosition ? { letter_position: s.letterPosition } : {}) };
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
      alt = modifyLetter(w, s.fromLetter, (chars, at) => { chars[at] = s.toLetter; }, s.letterPosition);
      // Replace the matching occurrence in the clean label too, not always
      // the first qaf when the word contains more than one.
      const before = lettersBefore(w, s.letterPosition, s.fromLetter);
      let occurrence = 0;
      m.said = [...m.expected].map(c => c === s.fromLetter && occurrence++ === before ? s.toLetter : c).join('');
      m.rule = 'letter_confusion';
      m.from_letter = s.fromLetter;
      m.to_letter = s.toLetter;
      instruction = { en: `In the highlighted word, say «${s.toLetter}» instead of «${s.fromLetter}» at letter position ${s.letterPosition}, exactly as shown. Keep the other letters and vowels unchanged.`, ar: `في الكلمة المظللة انطق «${s.toLetter}» بدل «${s.fromLetter}» في الموضع ${s.letterPosition} كما هو مكتوب، وأبقِ بقية الحروف والحركات كما هي.` };
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
    }, s.letterPosition);
    Object.assign(m, { said_uthmani: alt, letter: letterIndex, from: s.from, to: s.to, ...(s.continue ? { continuous_reading: true } : {}), ...(s.vowelSource ? { vowel_source: s.vowelSource } : {}) });
    mark.text = alt;
    instruction = { en: `Read as shown: on «${s.letter}» at letter position ${s.letterPosition} in the highlighted word, use ${s.to} instead of ${s.from}.${s.continue ? ' Continue directly into the next word without pausing, so the changed vowel is audible.' : ' Read the rest normally.'}`, ar: `اقرأ كما هو مكتوب: على «${s.letter}» في الموضع ${s.letterPosition} من الكلمة المظللة ${VOWEL_AR[s.to]} بدل ${VOWEL_AR[s.from]}.${s.continue ? ' أكمل إلى الكلمة التالية دون وقف، حتى تُسمع الحركة المتغيّرة.' : ' اقرأ الباقي كالمعتاد.'}` };
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
      modifyLetter(w, s.letter, (_, __, marks) => { if (!marks.includes('\u0651')) throw new Error(`Missing doubled letter: ${s.id}`); }, s.letterPosition);
      instruction = { en: `Keep the doubled ${s.letter === 'م' ? 'meem' : 'noon'} at letter position ${s.letterPosition} in the highlighted word, but omit its sustained nasal hum (ghunna). Read everything else normally.`, ar: `أبقِ تشديد ${s.letter === 'م' ? 'الميم' : 'النون'} في الموضع ${s.letterPosition} من الكلمة المظللة، لكن دون إطالة الغنة. اقرأ الباقي كالمعتاد.` };
    } else if (s.rule === 'light_emphatic') {
      if (!w.includes(s.letter)) throw new Error(`Missing emphatic letter: ${s.id}`);
      m.letter = s.letter;
      instruction = { en: `In the highlighted word, read «${s.letter}» at letter position ${s.letterPosition} too lightly, without its usual heavy resonance. Keep the same letter and read the rest normally.`, ar: `في الكلمة المظللة اقرأ «${s.letter}» في الموضع ${s.letterPosition} بترقيق بدل التفخيم، مع إبقاء الحرف نفسه. اقرأ الباقي كالمعتاد.` };
    } else if (s.rule === 'no_shadda') {
      const alt = modifyLetter(w, s.letter, (chars, at, marks) => {
        const offset = marks.indexOf('\u0651');
        if (offset < 0) throw new Error(`Missing shadda: ${s.id}`);
        chars.splice(at + 1 + offset, 1);
      }, s.letterPosition);
      m.said_uthmani = mark.text = alt;
      instruction = { en: `Read as shown: pronounce «${s.letter}» at letter position ${s.letterPosition} in the highlighted word once, without doubling it. Keep the other letters and vowels normal.`, ar: `اقرأ كما هو مكتوب: انطق «${s.letter}» في الموضع ${s.letterPosition} من حروف الكلمة المظللة مرة واحدة دون تشديد، وأبقِ بقية الحروف والحركات كالمعتاد.` };
    }
  }
  label.mistakes = [m];
  return { v: 3, seed, label, instruction, display: [display(v, s.word, mark)] };
}
