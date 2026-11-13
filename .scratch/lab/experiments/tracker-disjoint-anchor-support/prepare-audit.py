"""Prepare diagnostic copies; optionally archive an unchanged git revision before uploading to Modal."""
from pathlib import Path
import argparse
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', help='Archive this unchanged engine revision locally; Modal needs no git checkout')
args = parser.parse_args()
root = Path.cwd()
out = root / '.scratch/lab/experiments/tracker-disjoint-anchor-support'
src = root / 'packages/core/src'

def imports(text):
    return re.sub(r'from "\./([^"/]+)\.js"', lambda m: f'from "../../../../packages/core/src/{m[1]}.js"', text)

if args.baseline:
    for name in ('local', 'tracker', 'checker'):
        text = subprocess.check_output(['git', 'show', f'{args.baseline}:packages/core/src/{name}.ts'], text=True)
        text = imports(text)
        if name == 'tracker':
            text = text.replace('"../../../../packages/core/src/local.js"', '"./local.baseline.js"')
        (out / f'{name}.baseline.ts').write_text(text)

tracker = imports((src / 'tracker.ts').read_text())
tracker = tracker.replace('let observed = "";', 'let auditTokens: DecodedPhoneme[] = [];\n  let observed = "";', 1)
tracker = tracker.replace('const observe = (ph: DecodedPhoneme) => {', 'const observe = (ph: DecodedPhoneme) => {\n    auditTokens.push(ph);')
tracker = tracker.replace('return emit(event, (from) => quietTrace(from, "locked", shortlist));', """
    if (event.timeMs >= Number(process.env.AUDIT_FROM_MS) && event.timeMs <= Number(process.env.AUDIT_TO_MS)) {
      const independent = event.skipSupport ? auditTokens.slice(event.skipSupport.destination.from, event.skipSupport.destination.to).map(p => p.p).join("") : "";
      const reference = index.phonemes.slice(pos.origin, pos.offset);
      console.log(JSON.stringify({ audit: "local-anchor", timeMs: event.timeMs, word, run: candidate.span.start,
        newRun, support: event.anchorSupport, witness: event.skipSupport, previous: last!.anchorSupport,
        independent, reference, cost: independent ? anchorFitCost(reference, independent) : null, pos }));
    }
    return emit(event, (from) => quietTrace(from, "locked", shortlist));""")
(out / 'tracker.audit.ts').write_text(tracker)

checker = imports((src / 'checker.ts').read_text())
checker = checker.replace('if (skipSupport.get(right) === false) return;', """if (skipSupport.get(right) === false) {
      console.log(JSON.stringify({ audit: "withheld-gap", left: words[ayahFirst[left]], right: words[ayahFirst[right]], timeMs: nowMs }));
      return;
    }""")
checker = checker.replace('ledgerDone.add(a);', """if (nowMs >= Number(process.env.AUDIT_FROM_MS) && nowMs <= Number(process.env.AUDIT_TO_MS)) console.log(JSON.stringify({ audit: "ledger", candidate: words[ayahFirst[a]], left: words[ayahFirst[left]], right: words[ayahFirst[right]], leftSpan, rightLock, rightRead, margin, rightTo, timeMs: nowMs }));
      ledgerDone.add(a);""")
(out / 'checker.audit.ts').write_text(checker)
