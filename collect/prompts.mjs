// Prompt generation. A prompt is a fully specified label: the reciter is told
// exactly which mistake to make, so the recording needs no labelling afterwards.
// Word indices are 1-based over `text_clean` words, which drops standalone
// waqf marks (ۚ ۖ ...) that appear as separate tokens in `text_uthmani`.

const FATHA = '\u064E', DAMMA = '\u064F', KASRA = '\u0650';
const SHORT_VOWELS = [FATHA, DAMMA, KASRA];
const VOWEL_NAME = { [FATHA]: 'fatha', [DAMMA]: 'damma', [KASRA]: 'kasra' };
const SHADDA = '\u0651', MADDAH = '\u0653';
const ARABIC_LETTER = /[\u0621-\u064A\u0671-\u06D3]/;
const MARK_ONLY = /^[\u06D6-\u06ED\u06DE\u06E9\s]+$/;

export const KIND_WEIGHTS = {
  none: 20,
  skip_word: 15,
  substitution: 15,
  repeat: 10,
  vowel: 15,
  tajweed: 10,
  skip_ayah: 15,
};

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pick = (rng, xs) => xs[Math.floor(rng() * xs.length)];

function pickWeighted(rng, weights) {
  const entries = Object.entries(weights);
  let r = rng() * entries.reduce((s, [, w]) => s + w, 0);
  for (const [k, w] of entries) if ((r -= w) < 0) return k;
  return entries[entries.length - 1][0];
}

export function loadQuran(verses) {
  const byRef = new Map();
  const list = verses.map(v => {
    const uthmani = v.text_uthmani.replace(/^\uFEFF/, '').split(/\s+/).filter(Boolean);
    const words = uthmani.filter(w => !MARK_ONLY.test(w));
    const clean = v.text_clean.replace(/^\uFEFF/, '').split(/\s+/).filter(Boolean);
    const verse = { surah: v.surah, ayah: v.ayah, words, clean, surahName: v.surah_name_en };
    byRef.set(`${v.surah}:${v.ayah}`, verse);
    return verse;
  });
  return { list, byRef };
}

// Familiar surahs are recited more naturally, so half the prompts come from them.
const FAMILIAR = new Set([1, 18, 36, 55, 56, 67, ...Array.from({ length: 36 }, (_, i) => 78 + i)]);

function candidateVerses(quran, rng, minWords, maxWords) {
  const pool = quran.list.filter(v => v.words.length >= minWords && v.words.length <= maxWords && v.words.length === v.clean.length);
  const familiar = pool.filter(v => FAMILIAR.has(v.surah));
  return rng() < 0.5 && familiar.length ? familiar : pool;
}

function innerWordIndex(rng, n) {
  return n >= 3 ? 2 + Math.floor(rng() * (n - 2)) : 1 + Math.floor(rng() * n);
}

// Letters with a short vowel that is not on the final letter (the final vowel
// is dropped at waqf, so changing it is not a detectable mistake).
function vowelSites(word) {
  const chars = [...word];
  let lastLetter = -1;
  chars.forEach((c, i) => { if (ARABIC_LETTER.test(c)) lastLetter = i; });
  const sites = [];
  for (let i = 0; i < chars.length; i++) {
    if (!SHORT_VOWELS.includes(chars[i])) continue;
    let base = i - 1;
    while (base >= 0 && !ARABIC_LETTER.test(chars[base])) base--;
    if (base < 0 || base >= lastLetter || opensLongVowel(chars, i)) continue;
    sites.push({ index: i, base });
  }
  return sites;
}

// A short vowel followed by a madd letter (or dagger alif) is half of a long
// vowel; swapping it yields an unpronounceable spelling like هَٰذُا.
function opensLongVowel(chars, vowelIndex) {
  let j = vowelIndex + 1;
  while (j < chars.length && !ARABIC_LETTER.test(chars[j])) {
    if (chars[j] === '\u0670') return true;
    j++;
  }
  if (j >= chars.length || !/[اىويٰ]/.test(chars[j])) return false;
  const next = chars[j + 1] ?? '';
  return !/[\u064B-\u0652]/.test(next);
}

// Only mandatory madds: muttasil (hamza after the maddah in the same word) or
// lazim (shadda / sukun after it). Word-final maddah is munfasil, where a
// short reading is a valid riwaya variant, not a mistake.
function hasMandatoryMadd(word) {
  const i = word.indexOf(MADDAH);
  return i >= 0 && /[ءأإئؤ\u0651\u0652]/.test(word.slice(i + 1));
}

function tajweedSites(word) {
  const sites = [];
  if (hasMandatoryMadd(word)) sites.push('madd_short');
  if (/[نم][\u064E-\u0650]?\u0651/.test(word)) sites.push('no_ghunna');
  return sites;
}

function ayahDisplay(v, marks = {}) {
  return { ref: `${v.surah}:${v.ayah}`, surahName: v.surahName, words: v.words.map((text, i) => ({ text, ...(marks[i + 1] ?? {}) })) };
}

const INSTRUCTIONS = {
  none: () => ({ en: 'Read it correctly, the way you normally would.', ar: 'اقرأ الآية قراءة صحيحة كما تقرؤها عادةً.' }),
  skip_word: w => ({ en: 'Skip the highlighted word. Read everything else normally.', ar: `اترك الكلمة المظللة «${w}» واقرأ الباقي كالمعتاد.` }),
  substitution: (w, alt) => ({ en: 'Say the word below instead of the highlighted word.', ar: `قل «${alt}» بدلاً من «${w}».` }),
  repeat: w => ({ en: 'Say the highlighted word twice in a row, then continue.', ar: `كرر الكلمة المظللة «${w}» مرتين ثم أكمل.` }),
  vowel: (w, alt, from, to) => ({ en: `Read the highlighted word as written below (a ${from} becomes a ${to}).`, ar: `اقرأ الكلمة المظللة هكذا «${alt}».` }),
  madd_short: w => ({ en: 'In the highlighted word, cut the long madd short (2 counts or less).', ar: `في «${w}» اقصر المد (حركتان أو أقل).` }),
  no_ghunna: w => {
    const meem = /م[\u064E-\u0650]?\u0651/.test(w) && !/ن[\u064E-\u0650]?\u0651/.test(w);
    return {
      en: `In the highlighted word, read the doubled ${meem ? 'meem' : 'noon'} without ghunna (no nasal hum).`,
      ar: `في «${w}» اقرأ ${meem ? 'الميم' : 'النون'} المشددة بلا غنة.`,
    };
  },
  skip_ayah: ref => ({ en: `Read these ayahs but skip the crossed-out ayah ${ref} entirely.`, ar: `اقرأ هذه الآيات واترك الآية المشطوبة ${ref} كاملة.` }),
};

function build(kind, quran, rng) {
  if (kind === 'skip_ayah') {
    const starts = quran.list.filter((v, i) => {
      const run = quran.list.slice(i, i + 3);
      return run.length === 3 && run.every(r => r.surah === v.surah && r.words.length >= 2 && r.words.length <= 14);
    });
    const familiar = starts.filter(v => FAMILIAR.has(v.surah));
    const first = pick(rng, rng() < 0.5 && familiar.length ? familiar : starts);
    const i = quran.list.indexOf(first);
    const run = quran.list.slice(i, i + 3);
    const skipped = `${run[1].surah}:${run[1].ayah}`;
    return {
      label: { kind, ref: `${first.surah}:${first.ayah}-${run[2].surah}:${run[2].ayah}`, mistakes: [{ kind, ayah: skipped }] },
      display: run.map((v, j) => ({ ...ayahDisplay(v), ...(j === 1 ? { mark: 'skip_ayah' } : {}) })),
      instruction: INSTRUCTIONS.skip_ayah(skipped),
    };
  }

  for (let attempt = 0; attempt < 200; attempt++) {
    const v = pick(rng, candidateVerses(quran, rng, 4, 22));
    const ref = `${v.surah}:${v.ayah}`;
    const base = { kind, ref };

    if (kind === 'none') {
      return { label: { ...base, mistakes: [] }, display: [ayahDisplay(v)], instruction: INSTRUCTIONS.none() };
    }

    const word = innerWordIndex(rng, v.words.length);
    const w = v.words[word - 1];
    const m = { kind, ayah: ref, word, expected: v.clean[word - 1] };

    if (kind === 'skip_word' || kind === 'repeat') {
      return { label: { ...base, mistakes: [m] }, display: [ayahDisplay(v, { [word]: { mark: kind } })], instruction: INSTRUCTIONS[kind](w) };
    }

    if (kind === 'substitution') {
      const others = quran.list.filter(o => o.surah === v.surah && o !== v && o.words.length === o.clean.length);
      const o = pick(rng, others.length ? others : quran.list);
      const j = Math.floor(rng() * o.words.length);
      // Context-dependent marks (iqlab meem, idgham tanween) are wrong once the
      // word is moved; show it in its pausal form.
      const alt = o.words[j].replace(/[\u06D6-\u06ED]/g, '');
      if (!o.clean[j] || o.clean[j] === v.clean[word - 1] || v.clean.includes(o.clean[j])) continue;
      return {
        label: { ...base, mistakes: [{ ...m, said: o.clean[j], said_uthmani: alt, said_from: `${o.surah}:${o.ayah}:${j + 1}` }] },
        display: [ayahDisplay(v, { [word]: { mark: kind, alt } })],
        instruction: INSTRUCTIONS.substitution(w, alt),
      };
    }

    if (kind === 'vowel') {
      const sites = vowelSites(w);
      if (!sites.length) continue;
      const site = pick(rng, sites);
      const chars = [...w];
      const from = chars[site.index];
      const to = pick(rng, SHORT_VOWELS.filter(x => x !== from));
      chars[site.index] = to;
      const alt = chars.join('');
      const letterIndex = [...w].slice(0, site.base + 1).filter(c => ARABIC_LETTER.test(c)).length;
      return {
        label: { ...base, mistakes: [{ ...m, said_uthmani: alt, letter: letterIndex, from: VOWEL_NAME[from], to: VOWEL_NAME[to] }] },
        display: [ayahDisplay(v, { [word]: { mark: kind, alt, altLetter: site.base } })],
        instruction: INSTRUCTIONS.vowel(w, alt, VOWEL_NAME[from], VOWEL_NAME[to]),
      };
    }

    if (kind === 'tajweed') {
      const wordsWithSites = v.words.map((x, i) => ({ i: i + 1, sites: tajweedSites(x) })).filter(x => x.sites.length);
      if (!wordsWithSites.length) continue;
      const chosen = pick(rng, wordsWithSites);
      const rule = pick(rng, chosen.sites);
      const cw = v.words[chosen.i - 1];
      return {
        label: { ...base, mistakes: [{ kind, rule, ayah: ref, word: chosen.i, expected: v.clean[chosen.i - 1] }] },
        display: [ayahDisplay(v, { [chosen.i]: { mark: kind, rule } })],
        instruction: INSTRUCTIONS[rule](cw),
      };
    }
  }
  return build('none', quran, rng);
}

export function generatePrompt(quran, seed, kind) {
  const rng = mulberry32(seed);
  const k = kind ?? pickWeighted(rng, KIND_WEIGHTS);
  const p = build(k, quran, rng);
  return { v: 1, seed, ...p };
}
