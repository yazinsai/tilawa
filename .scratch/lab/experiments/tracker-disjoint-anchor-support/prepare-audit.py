"""Create read-only diagnostic copies; never change the shipped matching recurrence."""
from pathlib import Path
import re

root = Path.cwd()
out = root / '.scratch/lab/experiments/tracker-disjoint-anchor-support'
src = root / 'packages/core/src'

def copied(name):
    text = (src / f'{name}.ts').read_text()
    return re.sub(r'from "\./([^"/]+)\.js"', lambda m: f'from "../../../../packages/core/src/{m[1]}.js"', text)

local = copied('local')
local = local.replace('step(p) {', 'step(p, auditIndex = observations) {')
local = local.replace('let origin = new Int32Array(0);', 'let origin = new Int32Array(0);\n  let support = new Int32Array(0);\n  let supportEnd = new Int32Array(0);\n  let observations = 0;')
local = local.replace('const o = new Int32Array(c.length);', 'const o = new Int32Array(c.length);\n    const sf = new Int32Array(c.length).fill(-1);\n    const st = new Int32Array(c.length);')
local = local.replace('o[i] = origin[old];', 'o[i] = origin[old];\n        sf[i] = support[old];\n        st[i] = supportEnd[old];')
local = local.replace('origin = o;', 'origin = o;\n    support = sf;\n    supportEnd = st;', 1)
local = local.replace('const o = new Int32Array(n);', 'const o = new Int32Array(n);\n      const sf = new Int32Array(n);\n      const st = new Int32Array(n);')
local = local.replace('let bo = origin[i];', 'let bo = origin[i];\n        let bs = support[i];\n        let bt = supportEnd[i];')
local = local.replace('bo = origin[i - 1];', 'bo = origin[i - 1];\n            bs = support[i - 1] < 0 ? auditIndex : support[i - 1];\n            bt = auditIndex + 1;')
local = local.replace('bo = from;', 'bo = from;\n              bs = auditIndex;\n              bt = auditIndex + 1;')
local = local.replace('bo = o[i - 1];', 'bo = o[i - 1];\n            bs = sf[i - 1];\n            bt = st[i - 1];')
local = local.replace('o[i] = bo;', 'o[i] = bo;\n        sf[i] = bs;\n        st[i] = bt;')
local = local.replace('origin = o;\n      edits = e;', 'origin = o;\n      support = sf;\n      supportEnd = st;\n      observations++;\n      edits = e;')
local = local.replace('return { offset: best,', 'return { supportFrom: support[i], supportTo: supportEnd[i], observations, offset: best,')
local = local.replace('save: () => ({ lo, cost, origin,', 'save: () => ({ support, supportEnd, observations, lo, cost, origin,')
local = local.replace('({ lo, cost, origin, edits, best, ayah } = s);', '({ lo, cost, origin, edits, best, ayah } = s);\n      ({ support, supportEnd, observations } = state as any);')
(out / 'local.audit.ts').write_text(local)

tracker = copied('tracker').replace('"../../../../packages/core/src/local.js"', '"./local.audit.js"')
tracker = tracker.replace('let observed = "";', 'let auditTokens: DecodedPhoneme[] = [];\n  let auditBase = 0;\n  let observed = "";', 1)
tracker = tracker.replace('const observe = (ph: DecodedPhoneme) => {', 'const observe = (ph: DecodedPhoneme) => {\n    auditTokens.push(ph);')
tracker = tracker.replace('local.step(ph.p);', 'local.step(ph.p, auditTokens.length - 1);')
tracker = tracker.replace('const startLocal = (m: Match) => {', '''const startLocal = (m: Match) => {
    auditBase = auditTokens.length;
    console.log(JSON.stringify({ audit: "global-anchor", timeMs: lastFrame * (frameMs ?? 0), offset: m.offset, length: m.length, from: observedFrames[0]?.start, to: observedFrames.at(-1)?.end, base: auditBase }));''')
tracker = tracker.replace('return emit(event, (from) => quietTrace(from, "locked", shortlist));', '''const a = pos as any;
    if (lastFrame * (frameMs ?? 0) >= Number(process.env.AUDIT_FROM_MS) && lastFrame * (frameMs ?? 0) <= Number(process.env.AUDIT_TO_MS)) console.log(JSON.stringify({ audit: "local-anchor", timeMs: event.timeMs, word, run: candidate.span.start, newRun, tokens: [a.supportFrom, a.supportTo], frames: [auditTokens[a.supportFrom]?.start, auditTokens[a.supportTo - 1]?.end], previous: prev, pos: a }));
    return emit(event, (from) => quietTrace(from, "locked", shortlist));''')
(out / 'tracker.audit.ts').write_text(tracker)

checker = copied('checker')
checker = checker.replace('ledgerDone.add(a);', '''if (obs.at(-1)!.end * frameMs >= Number(process.env.AUDIT_FROM_MS) && obs.at(-1)!.end * frameMs <= Number(process.env.AUDIT_TO_MS)) console.log(JSON.stringify({ audit: "ledger", candidate: words[ayahFirst[a]], left: words[ayahFirst[left]], right: words[ayahFirst[right]], leftSpan, rightLock, rightRead, margin, rightTo, timeMs: obs.at(-1)!.end * frameMs }));
      ledgerDone.add(a);''')
(out / 'checker.audit.ts').write_text(checker)
