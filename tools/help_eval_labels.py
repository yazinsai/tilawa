"""Snapshot held-out labels for a provisional eval without changing the ongoing audit.

Only reviewed-and-approved prompt swaps are applied. Audited bad/unsure clips and
unreviewed known swaps are excluded. The original speaker split is preserved.
"""
import json
from pathlib import Path

from help_import import QURAN, apply_swaps, label

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / ".scratch/help"


def main():
    original = {r["id"]: r for r in map(json.loads, (DIR / "labels.jsonl").read_text().splitlines())}
    answers = {r["id"]: r for r in map(json.loads, (DIR / "answers.jsonl").read_text().splitlines())}
    swaps = json.loads((DIR / "version_swap.json").read_text())
    passage = json.loads((DIR / "passage.json").read_text())
    ref = {f"{a['s']}:{a['a']}": a["w"] for a in json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())["ayahs"]}
    q = {f"{a['surah']}:{a['ayah']}": a for a in json.loads(QURAN.read_text(encoding="utf-8-sig"))}
    rows = []
    for raw in map(json.loads, (DIR / "raw/submissions.jsonl").read_text().splitlines()):
        old = original[raw["id"]]
        if old["split"] != "eval":
            continue
        verdict = answers.get(raw["id"], {}).get("verdict")
        reason = None
        if verdict is not None and verdict != "1":
            reason = f"audit_verdict_{verdict}"
        elif raw["id"] in swaps and verdict != "1":
            reason = "unreviewed_version_swap"
        elif verdict is None and (p := passage.get(raw["id"])) and (p["wrong_passage"] or p["per_expected"] > 0.5):
            reason = "unaudited_teacher_mismatch"
        elif verdict is None and raw.get("extra_mistake"):
            reason = "unaudited_extra_mistake"
        if raw["id"] in swaps and verdict == "1":
            raw = apply_swaps([raw], swaps)[0]
            old = {**old, **label(q, ref, raw), "ayahs": raw.get("ayahs") or [raw["ref"]]}
            # Ranges need the same expansion as the importer.
            from help_import import ayahs_of
            old["ayahs"] = ayahs_of(raw)
        if not old["scored"]:
            reason = reason or "import_unscored"
        # Letter-confusion substitutions are pronunciation positives, not wrong-word positives.
        pronunciation_pair = any(m.get("rule") == "letter_confusion" for m in raw.get("mistakes", []))
        rows.append({**old, "audit": "verified" if verdict == "1" else "provisional", "exclude": reason,
                     "pronunciationPair": pronunciation_pair})
    out = DIR / "eval-labels.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} held-out labels; {sum(r['exclude'] is not None for r in rows)} excluded; "
          f"{sum(r['audit'] == 'verified' and r['exclude'] is None for r in rows)} verified usable -> {out}")


if __name__ == "__main__":
    main()
