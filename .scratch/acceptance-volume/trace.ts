import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadReferenceIndex } from '../../packages/core/src/reference.js';
import { createChecker } from '../../packages/core/src/checker.js';
import { chunks } from '../../packages/core/src/bench/lab.js';

// Instrument a disposable source copy; the worktree's engine and scorer remain untouched.
const core = fileURLToPath(new URL('../../packages/core/src/',import.meta.url));
const source = readFileSync(`${core}tracker.ts`,'utf8')
  .replace(/from "\.\//g,`from "${core}`)
  .replace('local.step(ph.p);','local.step(ph.p);\n      (globalThis as any).localAudit?.(ph, before, local.save());');
const audit = `${process.env.LAB_OUT}/audit-tracker.ts`;
writeFileSync(audit,source);
const {createTracker,trackerInput} = await import(audit);
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL('../../packages/core/assets/ref-v2.json',import.meta.url), 'utf8')));
const phrases = JSON.parse(readFileSync(new URL('../../packages/core/assets/phrases-v1.json',import.meta.url), 'utf8'));
const session = process.argv[2];
const listener = process.argv[3];
const at = Number(process.argv[6]);
if (!Number.isFinite(at)) throw new Error('Expected session, listener, two word references and time in seconds');
const ev = JSON.parse(readFileSync(`${process.env.LAB_DATA}/lab-v1.5u/sessions/${session}.${listener}.phone.evidence.json`, 'utf8'));
const key = (w: any) => `${w.surah}:${w.ayah}:${w.word}`;
const words = process.argv.slice(4,6).map(k => index.words.find(w => key(w) === k)!);
let history: any[] = [];
const snapshots: any[] = [];
(globalThis as any).localAudit = (ph: any, before: any, after: any) => {
  history.push(ph);
  const time = ph.start * ev.frameMs / 1000;
  if (time < at-2 || time > at+4) return;
  const candidates = words.map(w => {
    const states = [];
    for (let at = w.start; at <= w.end; at++) {
      const i = at - after.lo;
      states.push({ at, cost: after.cost[i], origin: after.origin[i], prefix: index.phonemes.slice(w.start, at) });
    }
    return { word: key(w), pronunciation: index.phonemes.slice(w.start,w.end), states };
  });
  snapshots.push({ time, p: ph.p, beforeBest: before.best, best: after.best,
    word: key(index.words[index.wordAt(after.best-1)]), candidates });
};
const tracker = createTracker(index, { phrases });
const checker = createChecker(index, { phrases });
const positions: any[] = [], checked: any[] = [];
for (const e of chunks(ev)) {
  const p = tracker.push(trackerInput(e)); positions.push(...p);
  checked.push(...checker.push(e,p));
}
const end = tracker.end(); positions.push(...end); checked.push(...checker.end(end));
writeFileSync(`${process.env.LAB_OUT}/target.json`, JSON.stringify({ snapshots, history, positions, checked },null,1));
console.log('snapshots', JSON.stringify(snapshots.filter(s => Math.abs(s.time-at)<=0.25)));
console.log('near positions',JSON.stringify(positions.filter(e => e.timeMs >=(at-2)*1000 && e.timeMs <=(at+4)*1000)));
console.log('confirmed',JSON.stringify(checked.filter(e => e.type==='checker' && e.state==='confirmed')));
