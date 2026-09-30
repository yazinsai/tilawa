// Print exact coverage for the corpus in use; optionally pass a corpus path.
import { readFile } from 'node:fs/promises';
import { loadQuran, diversitySummary } from '../server/help/prompts.mjs';
const path = process.argv[2] ?? new URL('../public/quran.json', import.meta.url);
console.log(JSON.stringify(diversitySummary(loadQuran(JSON.parse(await readFile(path, 'utf8')))), null, 2));
