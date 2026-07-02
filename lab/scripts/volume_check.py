"""Quick check for hq run results."""
import modal

app = modal.App("volume-check")
vol = modal.Volume.from_name("fastconformer-phoneme-training", create_if_missing=True)


@app.function(volumes={"/training": vol}, timeout=120)
def check_run():
    from pathlib import Path
    import json

    run = Path("/training/fastconformer-phoneme-v4-tlog-hq")
    if not run.exists():
        print("hq dir does not exist")
        return

    for d in sorted(run.iterdir()):
        if d.is_dir():
            children = list(d.iterdir())
            print(f"  {d.name}: {len(children)} items")
            for c in children[:5]:
                if c.is_file():
                    print(f"    {c.name}: {c.stat().st_size / 1e6:.1f} MB")
                elif c.is_dir():
                    print(f"    {c.name}/ (dir)")
        elif d.is_file():
            print(f"  {d.name}: {d.stat().st_size / 1e6:.1f} MB")

    model_path = run / "model" / "model.nemo"
    print(f"\nmodel.nemo exists: {model_path.exists()}")
    if model_path.exists():
        print(f"model.nemo size: {model_path.stat().st_size / 1e9:.2f} GB")

    meta = run / "model" / "training_metadata.json"
    if meta.exists():
        print(f"\ntraining_metadata.json:")
        print(meta.read_text())

    data_meta = run / "manifests" / "data_metadata.json"
    if data_meta.exists():
        print(f"\ndata_metadata.json:")
        print(data_meta.read_text())

    quality = run / "manifests" / "tlog_quality_report.json"
    if quality.exists():
        report = json.loads(quality.read_text())
        summary = report.get("summary", {})
        print(f"\nTLOG quality summary: {json.dumps(summary, indent=2)}")


@app.local_entrypoint()
def main():
    check_run.remote()
