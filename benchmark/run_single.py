"""Run a single experiment on both corpora, save to persistent file."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.runner import discover_experiments, run_experiment
from shared.streaming import StreamingPipeline
from shared.quran_db import QuranDB
from pathlib import Path

exp_filter = sys.argv[1] if len(sys.argv) > 1 else None
if not exp_filter:
    print("Usage: python benchmark/run_single.py <experiment-name>")
    sys.exit(1)

safe_name = exp_filter.replace('/', '__')
RESULTS_DIR = Path('benchmark/experiment_results')
RESULTS_FILE = RESULTS_DIR / f'{safe_name}.json'

results = {}
db = QuranDB()
pipeline = StreamingPipeline(db=db)

import benchmark.runner as br

for corpus_name in ['test_corpus', 'test_corpus_v2']:
    corpus_dir = Path('benchmark') / corpus_name
    br.CORPUS_DIR = corpus_dir

    manifest_path = corpus_dir / "manifest.json"
    with open(manifest_path) as f:
        samples = json.load(f)["samples"]

    experiments = discover_experiments(exp_filter)

    for exp in experiments:
        key = f"{exp['name']}|{corpus_name}"
        print(f">>> {exp['name']} on {corpus_name} ({len(samples)} samples)...", flush=True)
        try:
            result = run_experiment(exp, samples, pipeline, mode='full')
            if result is None:
                print(f"  SKIPPED", flush=True)
                results[key] = {'experiment': exp['name'], 'corpus': corpus_name, 'error': 'no transcribe/predict'}
                continue
            results[key] = {
                'experiment': exp['name'],
                'corpus': corpus_name,
                'recall': round(result['recall'], 4),
                'precision': round(result['precision'], 4),
                'sequence_accuracy': round(result['sequence_accuracy'], 4),
                'avg_latency': round(result['avg_latency'], 3),
                'model_size': result['model_size'],
                'total': result['total'],
            }
            r = results[key]
            print(f"  R={r['recall']:.0%} P={r['precision']:.0%} SA={r['sequence_accuracy']:.0%} L={r['avg_latency']:.2f}s", flush=True)
        except Exception as e:
            err_msg = str(e)[:200]
            print(f"  ERROR: {err_msg}", flush=True)
            results[key] = {'experiment': exp['name'], 'corpus': corpus_name, 'error': err_msg}

    # Save after each corpus (incremental)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

print(f"Saved to {RESULTS_FILE}", flush=True)
