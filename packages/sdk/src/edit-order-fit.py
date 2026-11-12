"""Fit the compact recurrent edit scorer and its shuffled control using training speakers only.

Called after edit-order-examples.ts. No practice/exam inputs are accepted by this estimator.
The hidden state is the preceding categorical operation; its sufficient features are the
25 ordered transition counts, centered on count-preserving uniform permutations.
"""
import argparse
import json
from pathlib import Path

import torch


def prepare(pairs):
    examples = []
    for pair in pairs:
        counts = pair["counts"]
        if len(counts) != 5 or counts[0] + counts[1] + counts[3] != pair["referenceLength"]:
            raise ValueError("Invalid count/length matching")
        for arm in ("positive", "negative"):
            for script in pair[arm]:
                actual = [script["sequence"].count(op) for op in ("match", "sub", "ins", "del", "boundary")]
                if actual != counts:
                    raise ValueError("Every ambiguous script must share the matched counts")
        features, weights = [], []
        for arm in ("positive", "negative"):
            features.append(torch.tensor([s["features"] for s in pair[arm]], dtype=torch.float64))
            acoustic = torch.tensor([-3 * s["cost"] / 10 for s in pair[arm]], dtype=torch.float64)
            weights.append(torch.log_softmax(acoustic, dim=0))
        control = torch.tensor(pair["control"], dtype=torch.float64)
        examples.append((pair["speaker"], features, weights, control))
    return examples


def batch(examples, shuffled):
    width = max(max(len(f) for f in e[1]) for e in examples)
    features = torch.zeros((len(examples), 2, width, 25), dtype=torch.float64)
    acoustic = torch.full((len(examples), 2, width), -torch.inf, dtype=torch.float64)
    speaker_counts = {}
    for speaker, *_ in examples:
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    for i, (_, f, a, control) in enumerate(examples):
        for arm in range(2):
            if shuffled:
                # Shared count-preserving shuffle within a matched pair. Both arms still run
                # the same 25-parameter model; count information cannot distinguish the labels.
                features[i, arm, 0] = control
                acoustic[i, arm, 0] = 0
            else:
                features[i, arm, :len(f[arm])] = f[arm]
                acoustic[i, arm, :len(f[arm])] = a[arm]
    balanced = torch.tensor([1 / speaker_counts[e[0]] / len(speaker_counts) for e in examples], dtype=torch.float64)
    return features, acoustic, balanced


def differences(weights, tensors, scale=1.0):
    features, acoustic, _ = tensors
    # Runtime scales each script's log ratio before marginalizing ambiguous scripts.
    # Scaling their already-marginalized difference is a different observation model.
    logits = torch.logsumexp(features @ (scale * weights) + acoustic, dim=2)
    return logits[:, 0] - logits[:, 1]


def fit(examples, penalty, shuffled):
    parameters = torch.zeros(25, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([parameters], max_iter=80, line_search_fn="strong_wolfe")
    tensors = batch(examples, shuffled)
    def closure():
        optimizer.zero_grad()
        losses = torch.nn.functional.softplus(-differences(parameters, tensors))
        loss = (losses * tensors[2]).sum() + penalty * parameters.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    return parameters.detach()


def run(source, destination):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    data = json.loads(source.read_text())
    metadata = data["metadata"]
    if metadata.get("trainingOnly") is not True:
        raise ValueError("Training-only provenance required")
    examples = prepare(data["pairs"])
    folds = metadata["folds"]
    if len(set(e[0] for e in examples)) < 3:
        raise ValueError("At least three training speakers with matched contrasts required")
    selection = []
    for penalty in (0.01, 0.1, 1.0):
        predictions = {scale: [] for scale in (0.0, 0.25, 0.5, 1.0, 2.0)}
        for fold in sorted(set(folds.values())):
            train = [e for e in examples if folds[e[0]] != fold]
            validation = [e for e in examples if folds[e[0]] == fold]
            if not train or not validation:
                raise ValueError("Empty speaker-separated training partition")
            ordered = fit(train, penalty, False)
            shuffled = fit(train, penalty, True)
            ordered_validation = batch(validation, False)
            shuffled_validation = batch(validation, True)
            for scale in predictions:
                order_predictions = differences(ordered, ordered_validation, scale).tolist()
                shuffled_predictions = differences(shuffled, shuffled_validation, scale).tolist()
                for e, order_prediction, control_prediction in zip(validation, order_predictions, shuffled_predictions):
                    predictions[scale].append((e[0], order_prediction, control_prediction))
        for scale, rows in predictions.items():
            by_speaker = {}
            for speaker, ordered, shuffled in rows:
                by_speaker.setdefault(speaker, []).append((ordered, shuffled))
            loss = sum(sum(float(torch.nn.functional.softplus(torch.tensor(-d))) for d, _ in rows) / len(rows)
                       for rows in by_speaker.values()) / len(by_speaker)
            control_loss = sum(sum(float(torch.nn.functional.softplus(torch.tensor(-d))) for _, d in rows) / len(rows)
                               for rows in by_speaker.values()) / len(by_speaker)
            selection.append({"penalty": penalty, "scale": scale, "speakerLogLoss": loss,
                              "shuffledLogLoss": control_loss,
                              "validationSpeakers": len(by_speaker)})
    selected = min(selection, key=lambda x: (x["speakerLogLoss"], -x["penalty"], x["scale"]))
    ordered = fit(examples, selected["penalty"], False)
    shuffled = fit(examples, selected["penalty"], True)
    model = {"version": 1, "ordered": ordered.tolist(), "shuffled": shuffled.tolist(), "scale": selected["scale"]}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "edit-order-model.json").write_text(json.dumps(model) + "\n")
    report = {**metadata, "selection": selection, "selected": selected,
              "calibration": "speaker-balanced out-of-fold log loss; refit once on authorized training speakers",
              "control": "equal capacity and pairs; count-keyed shuffle removes original ordering before scoring",
              "limits": "16 k-shortest scripts, transcript-constrained latent crops, first-order recurrent state"}
    (destination / "edit-order-fit-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.examples, args.out)
