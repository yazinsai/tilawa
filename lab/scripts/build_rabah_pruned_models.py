"""Build local pruned Rabah CTC checkpoints for benchmarking.

Example:
  .venv/bin/python scripts/build_rabah_pruned_models.py \
    --layers 12 8 6 \
    --output-root data \
    --save-source
"""

import argparse
import copy
import io
import json
from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

DEFAULT_SOURCE = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final"


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _estimate_int8_size_bytes(model: torch.nn.Module) -> int | None:
    try:
        quantized = torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        buf = io.BytesIO()
        torch.save(quantized.state_dict(), buf)
        return buf.getbuffer().nbytes
    except Exception:
        return None


def _evenly_spaced_indices(total_layers: int, keep_layers: int) -> list[int]:
    if keep_layers >= total_layers:
        return list(range(total_layers))
    if keep_layers <= 1:
        return [total_layers - 1]

    raw = [
        round(i * (total_layers - 1) / (keep_layers - 1))
        for i in range(keep_layers)
    ]

    # Enforce strictly increasing, in-range indices.
    fixed = []
    prev = -1
    for idx in raw:
        idx = max(idx, prev + 1)
        fixed.append(min(idx, total_layers - 1))
        prev = fixed[-1]
    return fixed


def _prune_model(model: Wav2Vec2ForCTC, layers: int) -> tuple[Wav2Vec2ForCTC, list[int]]:
    pruned = copy.deepcopy(model)
    current_layers = len(pruned.wav2vec2.encoder.layers)
    if layers > current_layers:
        raise ValueError(f"Cannot keep {layers} layers; model has {current_layers}")

    keep_indices = _evenly_spaced_indices(current_layers, layers)
    pruned.wav2vec2.encoder.layers = torch.nn.ModuleList(
        [pruned.wav2vec2.encoder.layers[i] for i in keep_indices]
    )
    pruned.config.num_hidden_layers = layers
    pruned.wav2vec2.config.num_hidden_layers = layers
    return pruned, keep_indices


def main():
    parser = argparse.ArgumentParser(description="Build pruned Rabah CTC checkpoints")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="HF ID or local model path")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[12, 8, 6],
        help="Layer counts to keep",
    )
    parser.add_argument("--output-root", default="data", help="Output root directory")
    parser.add_argument(
        "--save-source",
        action="store_true",
        help="Also save full source checkpoint to data/rabah-ctc-source",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading source model from {args.source}...")
    processor = Wav2Vec2Processor.from_pretrained(args.source)
    model = Wav2Vec2ForCTC.from_pretrained(args.source)
    model.eval()

    report = {
        "source": args.source,
        "source_layers": len(model.wav2vec2.encoder.layers),
        "variants": {},
    }

    if args.save_source:
        source_dir = output_root / "rabah-ctc-source"
        source_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving full source checkpoint to {source_dir}...")
        model.save_pretrained(source_dir)
        processor.save_pretrained(source_dir)
        report["source_size_bytes"] = _dir_size_bytes(source_dir)

    for layers in args.layers:
        print(f"\nBuilding pruned checkpoint ({layers} layers)...")
        pruned, keep_indices = _prune_model(model, layers)

        out_dir = output_root / f"rabah-ctc-pruned-{layers}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pruned.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)

        fp32_size = _dir_size_bytes(out_dir)
        int8_size = _estimate_int8_size_bytes(pruned)

        report["variants"][str(layers)] = {
            "path": str(out_dir),
            "fp32_size_bytes": fp32_size,
            "int8_state_dict_size_bytes": int8_size,
            "layer_indices": keep_indices,
        }

        if int8_size is None:
            int8_label = "n/a (quantization backend unavailable)"
        else:
            int8_label = f"{int8_size / (1024 ** 2):.1f}MB"

        print(
            "Saved",
            out_dir,
            f"| fp32={fp32_size / (1024 ** 2):.1f}MB",
            f"| int8-est={int8_label}",
        )

    report_path = output_root / "rabah-ctc-pruned-report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
