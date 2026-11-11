"""Contradictory full targets within acoustically verified recording families.

No transcript is chosen as correct. Equivalences come from the existing canon
representation and the reference's connected/isolated pronunciations of one
word. Acoustic confusion costs are deliberately not pronunciation equivalences.
"""
import ast
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def canonicalizer():
    # Load only the pure helpers, avoiding the evaluation entry point's process
    # arguments, configured paths and optional analysis dependencies.
    path = Path(__file__).resolve().parents[1]/'tools/per_breakdown.py'
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('collapse', 'canon')]
    namespace = {'QALQALA': 'ڇ'}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['canon']


def pronunciation_equivalences(reference):
    """Direct same-word renderings already admitted by loadReferenceIndex.

    Keep pairs, never a transitive closure through other words. This uses the
    complete generic reference asset, without selecting passages or speakers.
    """
    canon = canonicalizer()
    pairs = set()
    for ayah in reference['ayahs']:
        for i, start in enumerate(ayah['o']):
            end = ayah['o'][i+1] if i+1 < len(ayah['o']) else len(ayah['p'])
            connected, isolated = ayah['p'][start:end], ayah['iso'][i]
            forms = {canon(connected), canon(isolated)}
            # Same plain-onset alternative as reference.ts, before canonical
            # run collapse removes the count that determines the prefix.
            def run(s):
                return len(s)-len(s.lstrip(s[0])) if s else 0
            extra = run(connected)-run(isolated) if connected and isolated and connected[0] == isolated[0] else 0
            if extra > 0:
                forms.add(canon(connected[extra:]))
            for a in forms:
                for b in forms:
                    if a and b and a != b:
                        pairs.add((a, b))
    return sorted(pairs)


class CompleteTargets:
    def __init__(self, equivalences=()):
        self.pairs = defaultdict(list)
        canon = canonicalizer()
        for a, b in equivalences:
            a, b = canon(a), canon(b)
            if a and b and a != b:
                self.pairs[a[0], b[0]].append((a, b))

    def compatible(self, left, right):
        """Both complete strings must be consumed; no free edges or edits."""
        a, b = canonicalizer()(left), canonicalizer()(right)
        if a == b:
            return True
        seen, pending = {(0, 0)}, [(0, 0)]
        for i, j in pending:
            if i == len(a) or j == len(b):
                if i == len(a) and j == len(b):
                    return True
                continue
            successors = []
            if a[i] == b[j]:
                successors.append((i+1, j+1))
            for x, y in self.pairs[a[i], b[j]]:
                if a.startswith(x, i) and b.startswith(y, j):
                    successors.append((i+len(x), j+len(y)))
            for position in successors:
                if position not in seen:
                    seen.add(position)
                    pending.append(position)
        return False


def contradictory_families(records, family_by_recording, targets):
    """Compare every pair in each verified family, independent of timing data."""
    by_family = defaultdict(list)
    for record in records:
        if not record['target_text']:
            raise ValueError('A full phone transcript is required')
        by_family[family_by_recording[record['key']]].append(record)
    contradictions = []
    for family, copies in sorted(by_family.items()):
        conflicts = []
        for i, a in enumerate(copies):
            for b in copies[i+1:]:
                if not targets.compatible(a['target_text'], b['target_text']):
                    conflicts.append(dict(left=a['key'], right=b['key'],
                                          left_target=a['target_text'], right_target=b['target_text']))
        if conflicts:
            contradictions.append(dict(family=family, recordings=[r['key'] for r in copies], conflicts=conflicts))
    return contradictions


def supervision_weights(views, family_by_recording, contradictions):
    """Exclude every copy and augmentation from CTC, KD and preservation loss.

    Both arms traverse the same minibatches and perform the same updates. The
    filtered objective's fixed population normalization accounts for exclusion;
    minibatch renormalization would change its meaning on small batches.
    """
    removed = {c['family'] for c in contradictions}
    raw = [float(v['original_weight']) for v in views]
    import math
    if not raw or any(not math.isfinite(w) or w <= 0 for w in raw):
        raise ValueError('Positive finite original weights required')
    keep = [0. if family_by_recording[v['recording']] in removed else w for v, w in zip(views, raw)]
    if not sum(keep):
        raise ValueError('No noncontradictory supervision remains')
    return ([w/sum(raw) for w in raw], [w/sum(keep) for w in keep])
