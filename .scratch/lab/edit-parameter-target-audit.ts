/** Read-only single-case replay; all session selection is supplied through CLI arguments. */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { createTracker, trackerInput } from '../../packages/core/src/tracker.js';
import { createChecker } from '../../packages/core/src/checker.js';
import { loadReferenceIndex } from '../../packages/core/src/reference.js';
import { chunks } from '../../packages/core/src/bench/lab.js';
import type { EditParameterPosterior } from '../../packages/core/src/edit-parameter.js';
import type { EngineEvent } from '../../packages/core/src/events.js';

const { values } = parseArgs({ options: {
  set: { type: 'string' }, session: { type: 'string' }, at: { type: 'string' },
  model: { type: 'string' }, mode: { type: 'string', default: 'audit' }, variant: { type: 'string', default: 'phone' },
} });
if (!values.set || !values.session || !values.model || !values.at) throw new Error('--set, --session, --model and --at are required');
if (!['audit', 'mean', 'marginal'].includes(values.mode!)) throw new Error('Unknown comparison mode');
const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error('LAB_DATA and LAB_OUT are required');
const posterior = JSON.parse(readFileSync(values.model, 'utf8')) as EditParameterPosterior;
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL('../../packages/core/assets/ref-v2.json', import.meta.url), 'utf8')));
const phrases = JSON.parse(readFileSync(new URL('../../packages/core/assets/phrases-v1.json', import.meta.url), 'utf8'));
const evidence = JSON.parse(readFileSync(join(data, values.set, 'sessions', `${values.session}.spliced-clean-int8.${values.variant}.evidence.json`), 'utf8'));
const comparisons: unknown[] = [], events: EngineEvent[] = [];
const at = Number(values.at);
let time = 0;
const word = (offset: number) => index.words[index.wordAt(Math.max(0, offset))];
const tracker = createTracker(index, { mode: 'free', phrases, localEditParameters: {
  posterior, mode: values.mode as 'audit' | 'mean' | 'marginal', onComparison(c) {
    if (Math.abs(time - at) > 5) return;
    comparisons.push({ time, previous: c.previous, weightedSelected: c.weightedSelected, decision: c.decision,
      trajectories: c.trajectories.map(p => ({ ...p, word: word(p.offset - 1), runStart: word(p.origin) })) });
  },
} });
const checker = createChecker(index, { phrases });
for (const e of chunks(evidence)) {
  time = (e.decodedTo ?? 0) * e.frameMs / 1000;
  const positions = tracker.push(trackerInput(e));
  events.push(...positions, ...checker.push(e, positions));
}
const final = tracker.end(); events.push(...final, ...checker.end(final));
const relevant = events.filter(e => (e.type === 'position' || e.type === 'checker')
  && (Math.abs(e.timeMs / 1000 - at) <= 10 || e.type === 'checker' && e.state === 'confirmed'));
writeFileSync(join(out, `edit-parameter-${values.mode}-audit.json`), JSON.stringify({ mode: values.mode, comparisons, events: relevant }, null, 2));
console.log(JSON.stringify({ mode: values.mode, comparisons: comparisons.length, events: relevant }));
