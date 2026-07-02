"""Evaluate phoneme CTC model on IqraEval test set.

Loads QuranMB.v2 test set, runs phoneme model, outputs CSV in IqraEval format.

Usage:
    python scripts/eval_iqraeval.py
    python scripts/eval_iqraeval.py --output results/iqraeval_submission.csv
    python scripts/eval_iqraeval.py --max-samples 10  # quick test
"""

import sys
import csv
import argparse
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Evaluate on IqraEval test set")
    parser.add_argument("--output", type=str, default="results/iqraeval_submission.csv")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples for testing (0=all)")
    args = parser.parse_args()

    from datasets import load_dataset, Audio
    import soundfile as sf

    # Import experiment
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "phoneme_run",
        str(PROJECT_ROOT / "experiments" / "fastconformer-phoneme" / "run.py"),
    )
    phoneme_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phoneme_mod)

    # Load test dataset
    print("Loading IqraEval/QuranMB.v2 test set...")
    test_ds = load_dataset("IqraEval/QuranMB.v2", split="test")
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))

    # Try loading references
    references = {}
    try:
        ref_ds = load_dataset("IqraEval/test_references", split="train")
        for sample in ref_ds:
            references[sample["ID"]] = sample.get("Labels", "")
        print(f"Loaded {len(references)} reference labels")
    except Exception as e:
        print(f"No test references available: {e}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total_per = 0.0
    evaluated = 0

    for idx, sample in enumerate(test_ds):
        if args.max_samples > 0 and idx >= args.max_samples:
            break

        sample_id = sample.get("ID", f"sample_{idx}")

        # Write audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            audio_array = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            sf.write(tmp_path, audio_array, sr)

            # Run phoneme model
            predicted = phoneme_mod.transcribe_phonemes(tmp_path)
            results.append({"ID": sample_id, "Labels": predicted})

            # Compute local metrics if references available
            if sample_id in references:
                from shared.phoneme_aligner import align_phoneme_strings
                alignment = align_phoneme_strings(predicted, references[sample_id])
                total_per += alignment["per"]
                evaluated += 1

            if (idx + 1) % 100 == 0:
                avg_per = total_per / evaluated if evaluated > 0 else 0
                print(f"Processed {idx + 1} samples (avg PER: {avg_per:.3f})")
        except Exception as e:
            print(f"Error on sample {sample_id}: {e}")
            results.append({"ID": sample_id, "Labels": ""})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Labels"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")
    print(f"Total samples: {len(results)}")
    if evaluated > 0:
        print(f"Average PER: {total_per / evaluated:.4f}")


if __name__ == "__main__":
    main()
