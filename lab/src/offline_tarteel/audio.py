import numpy as np
import librosa

TARGET_SR = 16000


def load_audio(path: str, sr: int = TARGET_SR) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)
