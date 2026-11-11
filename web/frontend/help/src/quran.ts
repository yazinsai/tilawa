// Display text by surah/ayah from the site's public quran.json (display only; the engine has its own reference).
export interface Ayah {
  surah: number;
  ayah: number;
  text: string;
  /** Words as displayed, 1-based like engine word refs (index 0 = word 1). Pause marks stay on their word. */
  words: string[];
}

export interface Surah {
  number: number;
  nameAr: string;
  nameEn: string;
  ayahs: number;
}

export interface QuranText {
  ayah(surah: number, ayah: number): Ayah | null;
  surahs: Surah[];
}

interface Row {
  surah: number;
  ayah: number;
  text_uthmani: string;
  surah_name: string;
  surah_name_en: string;
}

const LETTER = /[\u0621-\u064A\u066E-\u06D3\u06FA-\u06FF]/;

/** Splits on spaces and glues stand-alone pause marks (ۖ ۗ ۚ …) onto the preceding word. */
export function splitWords(text: string): string[] {
  const words: string[] = [];
  for (const tok of text.replace(/^\uFEFF/, "").trim().split(/\s+/)) {
    if (!tok) continue;
    if (!LETTER.test(tok) && words.length) words[words.length - 1] += ` ${tok}`;
    else words.push(tok);
  }
  return words;
}

export function indexQuran(rows: Row[]): QuranText {
  const byKey = new Map<string, Ayah>();
  const surahs: Surah[] = [];
  for (const r of rows) {
    const text = r.text_uthmani.replace(/^\uFEFF/, "");
    byKey.set(`${r.surah}:${r.ayah}`, { surah: r.surah, ayah: r.ayah, text, words: splitWords(text) });
    const s = surahs[r.surah - 1];
    if (s) s.ayahs = Math.max(s.ayahs, r.ayah);
    else surahs[r.surah - 1] = { number: r.surah, nameAr: r.surah_name, nameEn: r.surah_name_en, ayahs: r.ayah };
  }
  return { ayah: (s, a) => byKey.get(`${s}:${a}`) ?? null, surahs: surahs.filter(Boolean) };
}

let loading: Promise<QuranText> | null = null;

export function loadQuran(url = "/quran.json"): Promise<QuranText> {
  loading ??= fetch(url)
    .then((r) => {
      if (!r.ok) throw new Error(`quran.json: HTTP ${r.status}`);
      return r.json() as Promise<Row[]>;
    })
    .then(indexQuran)
    .catch((e) => {
      loading = null;
      throw e;
    });
  return loading;
}
