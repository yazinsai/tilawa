"""Acquire original continuous QUD benchmark audio for a fresh streaming-student replay.

uv run --python cpython-3.12-macos-aarch64-none --with huggingface-hub --with pyarrow --with 'numpy>=2' tools/session_qud_audio.py --ids jaber-inshiqaq --out .scratch/session-bench/qud

Uses normal Hugging Face credential storage if necessary. Never logs credentials.
Original bytes and waveform remain private/gitignored; only provenance is reported.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

REPO = 'QUD-Technologies/quran-alignment-benchmark'

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ids', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--revision', help='Pinned dataset commit; defaults to resolving the current revision once')
    a = ap.parse_args(); selected = set(a.ids.split(',')); a.out.mkdir(parents=True, exist_ok=True)
    revision = a.revision or HfApi().dataset_info(REPO).sha
    path = hf_hub_download(REPO, 'v1/test-00000-of-00001.parquet', repo_type='dataset', revision=revision)
    rows = pq.read_table(path).to_pylist()
    provenance = []
    for r in rows:
        if r['id'] not in selected: continue
        if r.get('riwayah') != 'hafs_an_asim': raise ValueError('Requested recording is not declared Hafs')
        data = r['audio']['bytes']; wav = a.out / f"{r['id']}.wav"
        with tempfile.NamedTemporaryFile(suffix='.audio') as tmp:
            tmp.write(data); tmp.flush()
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp.name, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(wav)], check=True)
        provenance.append({'id': r['id'], 'reciter': r['reciter'], 'repo': REPO, 'revision': revision,
                           'file': 'v1/test-00000-of-00001.parquet', 'originalSha256': hashlib.sha256(data).hexdigest(),
                           'wavSha256': hashlib.sha256(wav.read_bytes()).hexdigest(), 'sourceSplit': 'test',
                           'sourceDurationS': r['duration_s'], 'riwayah': r['riwayah'], 'continuousOriginal': True,
                           'limitation': 'Existing evaluation recording. Training reciter/source overlap is not independently ruled out; diagnostic only.'})
        print(f"Acquired continuous {r['id']} ({r['duration_s']:.1f} s)")
    if selected != {p['id'] for p in provenance}: raise ValueError('Requested ids absent from source')
    (a.out / 'provenance.json').write_text(json.dumps({'version': 1, 'recordings': provenance}, indent=2) + '\n')

if __name__ == '__main__': main()
