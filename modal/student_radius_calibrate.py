"""Freeze an additive-background radius rejection region on training speakers only.

Each leave-one-speaker-out fold picks the largest cutoff preserving every measured
true substitution in its other speakers. The final cutoff is the minimum of those
fold cutoffs and every held-out positive radius. Null search results are censored,
so neither a cutoff nor a positive preservation guarantee is inferred from them.
No practice input or target identifier is accepted by this utility.
"""
import argparse
import json
import math
from pathlib import Path


def calibrate(parts):
    if not parts or any(p.get("schema") != 1 for p in parts):
        raise ValueError("Missing training shard")
    shards = parts[0]["shards"]
    if sorted(p["shard"] for p in parts) != list(range(shards)):
        raise ValueError("Calibration requires exactly one of every training shard")
    excluded = set().union(*(set(p["excludedSpeakers"]) for p in parts))
    rows = [r for p in parts for r in p["rows"]]
    if any(r["split"] != "train" or r["speaker"] in excluded for r in rows):
        raise ValueError("Nontraining or excluded speaker in calibration")
    protocols = {(r["primarySha256"], r["inventory"], json.dumps(r["search"], sort_keys=True)) for r in rows}
    if len(protocols) != 1:
        raise ValueError("No accusations or mixed listener/search protocols")
    usable = [r for r in rows if r["result"]["status"] == "crossed" and
              isinstance(r["result"]["energy"], (int, float)) and math.isfinite(r["result"]["energy"]) and
              r["result"]["energy"] > 0]
    speakers = sorted({r["speaker"] for r in usable})
    positive_speakers = {r["speaker"] for r in usable if r["positive"]}
    if len(positive_speakers) < 2 or not any(not r["positive"] for r in usable):
        raise ValueError("Radius calibration lacks measurable true substitutions from two speakers or false accusations")
    folds = []
    for speaker in speakers:
        train_positive = [r["result"]["energy"] for r in usable if r["positive"] and r["speaker"] != speaker]
        if not train_positive:
            raise ValueError("A speaker fold lacks measured positive controls")
        cutoff = min(train_positive)
        validation = [r for r in usable if r["speaker"] == speaker]
        folds.append({"speaker": speaker, "cutoff": cutoff,
                      "trueLost": sum(r["positive"] and r["result"]["energy"] < cutoff for r in validation),
                      "falseVetoed": sum(not r["positive"] and r["result"]["energy"] < cutoff for r in validation)})
    cutoff = min([f["cutoff"] for f in folds] + [r["result"]["energy"] for r in usable if r["positive"]])
    won = sum(not r["positive"] and r["result"]["energy"] < cutoff for r in usable)
    if not won:
        raise ValueError("Training-frozen rejection region contains no measured false accusations")
    sha, inventory, search = next(iter(protocols))
    calibration = dict(schema=1, inventory=inventory, primarySha256=sha,
                       rejectionEnergy=cutoff, search=json.loads(search))
    report = dict(recordings=parts[0]["recordings"], accusations=len(rows), measured=len(usable),
                  positiveSpeakers=len(positive_speakers), folds=folds,
                  frozenFalseVetoed=won, frozenTrueLost=0,
                  censored=sum(r["result"]["status"] == "no-crossing" for r in rows),
                  abstained=sum(r["result"]["status"] in ("no-pause", "baseline-mismatch") for r in rows))
    return calibration, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model, report = calibrate([json.loads(Path(p).read_text()) for p in args.parts])
    Path(args.out).write_text(json.dumps(model, indent=2) + "\n")
    Path(args.out + ".validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
