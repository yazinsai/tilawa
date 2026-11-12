"""Prepare, select and export three matched frozen-head arms on training speakers.

Run only through the supervisor. Enrolled performed transcripts, explicit
release-speaker exclusions and shipped augmentation banks are required inputs;
there is no fallback to practice audio or mistake scripts. Exports retain the
native encoder and the installed SDK insertion teacher unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from student_head_data import (SEED, TOKENS, ctc_collapse, edit_distance, contained,
                               prepare, load_prepared, write_json)
from student_head_export import (digest, load_source, head_parameters, graft,
                                 insertion_assets, session, stream)
from student_negative_curvature import phone_head, log_probs, corpus_evaluate, matched_refit


def validate(head, rows, initial, regularization):
    """Equal selection speakers; all are disjoint from fit and release speakers."""
    speakers = {}
    for row in rows:
        objective, _, _ = corpus_evaluate(head, initial, [row], regularization)
        errors = []
        for view in row["views"]:
            hypothesis = ctc_collapse(log_probs(head, view["features"]).argmax(1).tolist())
            errors.append(edit_distance(hypothesis, view["target"].tolist()) / len(view["target"]))
        speakers.setdefault(row["speaker"], []).append([objective, float(np.mean(errors))])
    if not speakers:
        raise ValueError("Need identified selection speakers")
    objective, per = np.mean([np.mean(v, axis=0) for v in speakers.values()], axis=0)
    return dict(objective=float(objective), per=float(per), speakers=len(speakers))


def fit(args):
    prepared, out = Path(args.prepared), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fit directory must be empty")
    info, rows = load_prepared(prepared)
    _, graph, _, _ = load_source(info["source"])
    initial = phone_head(*head_parameters(graph))
    heads, report = matched_refit(initial, rows, info["groups"], validate, seed=SEED,
        seconds=args.seconds, emit=lambda r: print(json.dumps(r, allow_nan=False), flush=True))
    report.update(seed=SEED, tokens=TOKENS,
        objective="equal-recording-mean-views-complete-CTC-plus-full-KD-plus-shipped-KL-plus-L2",
        mechanism="exact-forward-backward-occupancy-sensitivity-HVP-and-actual-loss-decreasing-escape",
        head="existing-native-affine-weights-dequantized-in-all-three-arms",
        groups=info["groups"], excludedSpeakers=info["excludedSpeakers"],
        preparedSha256=digest(prepared / "prepared.json"), arms={})
    out.mkdir(parents=True, exist_ok=True)
    selection = [r for r in rows if info["groups"][r["speaker"]] == "selection"]
    for arm, head in heads.items():
        np.savez(out / (arm + ".npz"), weight=head.weight, bias=head.bias)
        path = graft(info["source"], head, out / arm, "negative-curvature-" + arm)
        runtime = session(path, optimize=True)
        manifest = json.loads((out / arm / "manifest.json").read_text())
        worst = 0.
        for row in selection:
            for file, view in zip(row["cacheFiles"], row["views"]):
                with np.load(contained(prepared, file), allow_pickle=False) as data:
                    got, _ = stream(runtime, manifest, data["fbank"])
                want = log_probs(head, view["features"])
                np.testing.assert_allclose(np.exp(got), np.exp(want), atol=2e-4, rtol=2e-4)
                worst = max(worst, float(np.abs(np.exp(got) - np.exp(want)).max()))
        report["arms"][arm] = dict(sha256=digest(path), parityMaxProbabilityError=worst,
            parityViews=3 * len(selection), auxiliaryHashes=insertion_assets(out / arm))
    # Neither a report nor a model declaration may claim a partial export valid.
    write_json(out / "training-report.json", report)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--help-labels", dest="help", required=True)
    for name in ("model", "transcribed", "exclusions", "augmentation", "out"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("fit")
    for name in ("prepared", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seconds", type=float, default=86400., help="Complete fitting deadline, at most one day")
    return parser


if __name__ == "__main__":
    args = argument_parser().parse_args()
    (prepare if args.command == "prepare" else fit)(args)
