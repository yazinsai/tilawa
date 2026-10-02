import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 16000


def _ffmpeg(path: str, sr: int) -> np.ndarray:
    import subprocess

    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    )
    audio = np.frombuffer(out.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg produced no samples for {path}")
    return audio.copy()


def load_audio(path: str, sr: int = TARGET_SR) -> np.ndarray:
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio.astype(np.float32)
    except Exception:
        pass
    try:
        audio, native_sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if native_sr != sr:
            audio = librosa.resample(audio, orig_sr=native_sr, target_sr=sr)
        return audio.astype(np.float32)
    except Exception:
        return _ffmpeg(path, sr)
