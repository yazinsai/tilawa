"""Verify each experiment exports a transcribe() function."""
import sys
import importlib.util
import inspect
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

EXPERIMENTS = {
    "ctc-alignment": ROOT / "experiments" / "ctc-alignment" / "run.py",
    "whisper-lora": ROOT / "experiments" / "whisper-lora" / "run.py",
    "tarteel-whisper-base": ROOT / "experiments" / "tarteel-whisper-base" / "run.py",
    "streaming-asr": ROOT / "experiments" / "streaming-asr" / "run.py",
}

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_all_experiments_have_transcribe():
    for name, path in EXPERIMENTS.items():
        if not path.exists():
            continue
        mod = _load(name, path)
        assert hasattr(mod, "transcribe"), f"{name} missing transcribe()"
        sig = inspect.signature(mod.transcribe)
        params = list(sig.parameters.keys())
        assert "audio_path" in params, f"{name}.transcribe() missing audio_path param"
