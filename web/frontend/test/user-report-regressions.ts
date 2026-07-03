import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { QuranDB } from "../src/lib/quran-db.ts";
import { RecitationTracker, type TranscribeResult } from "../src/lib/tracker.ts";
import { SAMPLE_RATE } from "../src/lib/types.ts";
import { adaptQuranTextData, type CtcTokenTable } from "../src/worker/quran-text-adapter.ts";
import { TextCTCDecoder } from "../src/worker/text-ctc-decode.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const REPORT_CORPUS = resolve(ROOT, "../../benchmark/user_reports_july");
const CHUNK_SAMPLES = SAMPLE_RATE * 2;

interface TranscriptCase {
  id: string;
  transcripts: string[];
  knownModelMiss?: string;
}

interface ManifestSample {
  id: string;
  expected_verses: { surah: number; ayah: number }[];
}

const CASES: TranscriptCase[] = [
  {
    id: "report_18_082_stuck_to_88",
    transcripts: [
      "وكان وراءهم ملك ياخذ كل سفينه غصبا واما الغلام فكان ابواه مؤمنين فخشينا ان يرهقهما طغيانا وكفرا",
      "كان ربهما خيرا منه زكاه واقرب رحما واما الجدتار فكان لغلامين يتيمين في المدينه وكان تحته كن",
      "قربحما واما الجدار فكان لغلامين يتيمين في المدينه وكان تحته كنز لهما وكان ابوهما صالحا فاراد ربك ان يبلغا اشدهما ويستخرجا كنزهما رحمه م ربك وماما فعله عن امرري ذلك تاويل ما لم تصطح",
      "اقرب رحماا واما ال الجدار فكان لغلامين يتيمين في المدينه وكان تحته كنز لهما وكان ابوهما صالحا فاراد ربك ان يبلغا اشدهما ويستخرجا كنزهما رحمه من ربك وما فعلته عن امري ذ تاويل ما لم تستطع عليه صبرا",
    ],
  },
  {
    id: "report_18_057_wrong_38_002",
    transcripts: [
      "وما يحسن المين",
      "وما نرسل المرسلين الا مبشرين",
      "وسلام على المرسلين",
      "يدحضوا به الحق واتخذوا اياتي وما انزرت",
      "ذكر بايات ربه فاعرض عنها ونسي ما قدمت يداه",
      "ذكر بايات ربه فاعرض عنها ونسي ما قدمت يداه انا جعلنا علي قلوبهم اكنه ان يفقهوه وفي اذانهم وقرا",
    ],
  },
  {
    id: "report_18_055_056_skip",
    transcripts: [
      "ولقد صرفنا في هذا القران للناس من كل مثل",
      "وما منع الناس ان يؤمنوا اذ جاءهم الهدي ويستغفروا ربهمم ال",
      "وما منع الناس ان يؤمنوا اذ جاءهم الهدي ويستغفروا ربهم الا ان تاتيهم سنه الاولين",
      "وما منع الناس ان يؤمنوا اذ جاءهم الهدي ويستغفروا ربهم الا ان تاتيهمم سنه الاولين او ياتيهم العذاب",
    ],
  },
  {
    id: "report_002_001_no_prediction",
    transcripts: [
      "الم ذلك الكتاب لنا",
      "الم ذلك الكتاب لا ريب فيه هدي للمين",
    ],
  },
  {
    id: "report_18_109_no_prediction",
    knownModelMiss:
      "Captured ASR transcript is Baqarah-like text, so matcher/tracker fixes cannot recover 18:109 from this decode.",
    transcripts: [
      "الم ذلك الكتاب لا ريب فيه هدي للمتق الذين يؤ بون",
    ],
  },
];

let db: QuranDB;

function verseKey(surah: number, ayah: number): string {
  return `${surah}:${ayah}`;
}

function makeSpeechChunk(): Float32Array {
  const audio = new Float32Array(CHUNK_SAMPLES);
  for (let i = 0; i < audio.length; i++) {
    audio[i] = 0.05 * Math.sin(i / 10);
  }
  return audio;
}

function dedupeOrdered(values: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    if (seen.has(value)) continue;
    seen.add(value);
    result.push(value);
  }
  return result;
}

function containsOrdered(expected: string[], actual: string[]): boolean {
  let cursor = 0;
  for (const value of actual) {
    if (value === expected[cursor]) cursor++;
    if (cursor === expected.length) return true;
  }
  return cursor === expected.length;
}

function createTranscriber(transcripts: string[]) {
  let callIndex = 0;
  const results = transcripts.map((text): TranscribeResult => {
    const champion = db.bestJoint03Match(text);
    return {
      text,
      rawPhonemes: text,
      tokenIds: [],
      championMatch: champion && champion.score >= 0.8 ? champion : undefined,
    };
  });

  return async (): Promise<TranscribeResult> => {
    const result = results[Math.min(callIndex, results.length - 1)];
    callIndex++;
    return result;
  };
}

function loadExpectedVerses(): Map<string, string[]> {
  const manifest = JSON.parse(
    readFileSync(resolve(REPORT_CORPUS, "manifest.json"), "utf-8"),
  ) as { samples: ManifestSample[] };
  return new Map(
    manifest.samples.map((sample) => [
      sample.id,
      sample.expected_verses.map((verse) => verseKey(verse.surah, verse.ayah)),
    ]),
  );
}

async function runCase(testCase: TranscriptCase): Promise<string[]> {
  const tracker = new RecitationTracker(db, createTranscriber(testCase.transcripts));
  const speech = makeSpeechChunk();
  const silence = new Float32Array(CHUNK_SAMPLES);
  const matches: string[] = [];

  for (let i = 0; i < testCase.transcripts.length + 6; i++) {
    for (const msg of await tracker.feed(speech)) {
      if (msg.type === "verse_match") matches.push(verseKey(msg.surah, msg.ayah));
    }
  }

  for (let i = 0; i < 4; i++) {
    for (const msg of await tracker.feed(silence)) {
      if (msg.type === "verse_match") matches.push(verseKey(msg.surah, msg.ayah));
    }
  }

  return dedupeOrdered(matches);
}

async function main() {
  const metadata = JSON.parse(readFileSync(resolve(ROOT, "public/export_metadata.json"), "utf-8"));
  const vocabJson = JSON.parse(readFileSync(resolve(ROOT, "public/vocab.json"), "utf-8"));
  const decoder = new TextCTCDecoder(vocabJson, Number(metadata.blank_id ?? 1024));
  const ctcTokens = JSON.parse(
    readFileSync(resolve(ROOT, "public/quran_ctc_tokens.json"), "utf-8"),
  ) as CtcTokenTable;
  const quranRaw = JSON.parse(readFileSync(resolve(ROOT, "public/quran.json"), "utf-8"));
  db = new QuranDB(adaptQuranTextData(quranRaw, ctcTokens, decoder), undefined, ctcTokens);
  const expectedByCase = loadExpectedVerses();

  let failures = 0;
  for (const testCase of CASES) {
    if (testCase.knownModelMiss) {
      console.log(`SKIP ${testCase.id} ${testCase.knownModelMiss}`);
      continue;
    }

    const expected = expectedByCase.get(testCase.id);
    if (!expected) {
      throw new Error(`Missing manifest sample for ${testCase.id}`);
    }

    const actual = await runCase(testCase);
    const passed = containsOrdered(expected, actual);
    if (!passed) failures++;
    console.log(
      `${passed ? "PASS" : "FAIL"} ${testCase.id} expected=[${expected.join(", ")}] got=[${actual.join(", ")}]`,
    );
  }

  if (failures > 0) {
    throw new Error(`${failures} user report regression case(s) failed`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
