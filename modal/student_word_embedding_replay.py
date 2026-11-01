"""Evaluate a calibrated word embedding on queries exported by the checker.

No fitting occurs here. Query audio comes from anchor/decoded boundaries; labels
are retained in the report solely to assess the preregistered kill criterion.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from student_word_embedding import FrozenMiddle, WordEncoder, acoustic_features


def folded_gap_changed(base, folded, gap):
    """Existing evidence predates provenance tags; compare the two gap paths.

    The only changed evidence in the folded listener is its isolated gap pass.
    Changes elsewhere in a recording cannot make this gap eligible.
    """
    if any(base[k] != folded[k] for k in ("inventory", "frameMs", "nFrames")):
        raise ValueError("Gap provenance requires matching listener frame clocks")
    a, b = gap["fromFrame"], gap["toFrame"]
    left, right = base["posteriors"], folded["posteriors"]
    if left["k"] != right["k"]:
        raise ValueError("Gap provenance requires matching posterior widths")
    k = left["k"]
    if any(left[key][a*k:b*k] != right[key][a*k:b*k] for key in ("ids", "probs")):
        return True
    phones = lambda evidence: [p for p in evidence["phonemes"] if a <= (p["start"] + p["end"]) / 2 < b]
    return phones(base) != phones(folded)


def trim_gap(wave):
    """Remove edge silence, keeping the same 80 ms margin as real-word crops.

    A fixed 40 dB range below this crop's peak frame power preserves weak speech.
    This is ordinary audio endpointing, independent of reference and labels.
    """
    if len(wave) < 400:
        return wave
    energy = np.array([np.mean(wave[a:a+400].astype(np.float64) ** 2) for a in range(0, len(wave) - 399, 160)])
    active = np.flatnonzero(energy > max(1e-10, float(energy.max()) * 1e-4))
    if not len(active):
        return wave[:0]
    lo = max(0, int(active[0]) * 160 - 1280)
    hi = min(len(wave), int(active[-1]) * 160 + 400 + 1280)
    return wave[lo:hi]


def compare(z, artifact, neighbours):
    distances = 1 - artifact["bank"] @ z.cpu()
    neighbour_ids = [i for i, key in enumerate(artifact["keys"]) if key in neighbours]
    missing = sorted(set(neighbours) - set(artifact["keys"]))
    winner = int(distances.argmin())
    whole_distance = float(distances[winner])
    neighbour_distance = float(distances[neighbour_ids].min()) if neighbour_ids else float("inf")
    prefix_ids = [i for i, k in enumerate(artifact["prefix_keys"]) if k in neighbour_ids]
    if prefix_ids:
        neighbour_distance = min(neighbour_distance, float((1 - artifact["prefix_bank"][prefix_ids] @ z.cpu()).min()))
    # Missing real exemplars cannot stand in for a tested repeat explanation.
    accept = (not missing and artifact["keys"][winner] not in neighbours
              and whole_distance < neighbour_distance and whole_distance <= artifact["report"]["radius"])
    top = distances.topk(min(5, len(distances)), largest=False)
    return dict(accept=bool(accept), winner=artifact["keys"][winner], distance=whole_distance,
                neighbour_distance=neighbour_distance if np.isfinite(neighbour_distance) else None,
                radius=artifact["report"]["radius"], missing_neighbours=missing,
                top=[dict(word=artifact["keys"][int(i)], distance=float(d)) for i, d in zip(top.indices, top.values)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--listener", required=True)
    parser.add_argument("--base-listener", required=True)
    parser.add_argument("--checkpoint", default="/student/prior-runs/spliced-clean/ckpt.pt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    artifact = torch.load(args.embedding, map_location="cpu", weights_only=False)
    if artifact["report"].get("prefix_method") != "independently encoded real prefixes":
        raise ValueError("Rebuild the reference prefix bank before evaluating additions")
    frozen = FrozenMiddle(args.checkpoint).cuda()
    model = WordEncoder(artifact["channels"]).cuda().eval()
    model.load_state_dict(artifact["model"])
    results = []
    for path in sorted(Path(args.queries).glob("*.json")):
        item = json.loads(path.read_text())
        audio_path = Path(args.audio) / f'{item["session"]}.{item["variant"]}.wav'
        rows = []
        if item["queries"]:
            wave, sr = sf.read(audio_path, dtype="float32")
            assert sr == 16000 and wave.ndim == 1
            evidence_path = lambda listener: Path(args.audio) / f'{item["session"]}.{listener}.{item["variant"]}.evidence.json'
            base = json.loads(evidence_path(args.base_listener).read_text())
            folded = json.loads(evidence_path(args.listener).read_text())
        for query in item["queries"]:
            # The anchored gap includes sounds omitted by greedy, including the
            # low-confidence ending of a word whose onset alone was decoded.
            if not query["gaps"]:
                rows.append(dict(**query, accept=False, reason="no anchored gap"))
                continue
            gap = min(query["gaps"], key=lambda g: g["toFrame"] - g["fromFrame"])
            if not folded_gap_changed(base, folded, gap):
                rows.append(dict(**query, accept=False, reason="not changed by folded gap path"))
                continue
            frame_s = query["frameMs"] / 1000
            crop = trim_gap(wave[int(gap["fromFrame"] * frame_s * sr):int(gap["toFrame"] * frame_s * sr)])
            if not 3840 <= len(crop) <= 4 * sr:
                rows.append(dict(**query, accept=False, reason="not a single-word duration"))
                continue
            feats = acoustic_features(crop, frozen.cfg)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pooled = frozen(feats[None].cuda(), torch.tensor([len(feats)], device="cuda"))
            with torch.no_grad():
                z = model(pooled.half().float(), torch.tensor([len(crop) / sr], device="cuda"))[0]
            rows.append(dict(**query, duration=len(crop) / sr, **compare(z, artifact, query["neighbours"])))
        result = dict(**item, results=rows)
        results.append(result)
        print(json.dumps(dict(session=item["session"], variant=item["variant"], clean=item["clean"],
                              queries=len(rows), confirmed=sum(r["accept"] for r in rows))), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    summary = dict(sessions=len(results), queries=sum(len(r["results"]) for r in results),
                   confirmed=sum(q["accept"] for r in results for q in r["results"]),
                   clean_confirmed=sum(q["accept"] for r in results if r["clean"] for q in r["results"]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
