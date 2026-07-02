"""Quick utility to inspect and clean the Modal training volume."""
import modal

app = modal.App("volume-cleanup")
vol = modal.Volume.from_name("fastconformer-phoneme-training", create_if_missing=True)


@app.function(volumes={"/training": vol}, timeout=600)
def list_volume():
    """List top-level dirs and their sizes."""
    from pathlib import Path

    root = Path("/training")
    for d in sorted(root.iterdir()):
        if d.is_dir():
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            print(f"{d.name}: {total / 1e9:.1f} GB")
            for sub in sorted(d.iterdir()):
                if sub.is_dir():
                    sub_total = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
                    print(f"  {sub.name}: {sub_total / 1e9:.1f} GB")


@app.function(volumes={"/training": vol}, timeout=1800)
def delete_dir(name: str):
    """Delete a directory path from the volume (supports nested like 'v3/audio')."""
    from pathlib import Path
    import shutil

    target = Path(f"/training/{name}")
    if not target.exists():
        print(f"{name} does not exist")
        return
    shutil.rmtree(target)
    vol.commit()
    print(f"Deleted {name}")


@app.local_entrypoint()
def main(delete: str = ""):
    if delete:
        for name in delete.split(","):
            delete_dir.remote(name.strip())
    else:
        list_volume.remote()
