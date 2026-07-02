# Offline Tarteel - Quran Verse Recognition

## Problem
Build an offline CLI tool that listens to Quran recitation audio and identifies the exact surah:ayah being recited. Must work with any reciter (not just matching against known recordings) and identify verses within ~10 seconds of audio.

## Approach: mlx-whisper STT + Fuzzy Text Matching

Use Apple's MLX framework with Tarteel AI's fine-tuned Whisper model to transcribe Arabic audio, then fuzzy-match the transcription against all 6,236 Quran verses.

### Why this approach
- **Reciter-independent**: Whisper understands Arabic speech regardless of the speaker
- **Proven model**: `tarteel-ai/whisper-base-ar-quran` achieves 5.75% WER on Quranic Arabic
- **Fast on Apple Silicon**: MLX uses Metal natively, near real-time inference
- **Small footprint**: ~290MB model + ~2MB text database

### Why NOT audio fingerprinting
- Chromaprint/AcoustID only matches exact recordings, not different reciters
- MFCC+DTW is too computationally expensive (comparing against 6,236+ references) and doesn't generalize across voices

## Architecture

```
Audio Input (file or mic)
    |
    v
Audio Preprocessor (16kHz mono, chunked ~10s)
    |
    v
mlx-whisper (tarteel-ai/whisper-base-ar-quran)
    |
    v
Arabic Transcription
    |
    v
Text Normalizer (strip diacritics, normalize Arabic chars)
    |
    v
Verse Matcher (n-gram surah narrowing -> Levenshtein ayah match)
    |
    v
Output: Surah Name : Ayah Number + Arabic text
```

## Components

### 1. Audio Input
- File mode: `python recognize.py audio.mp3`
- Mic mode: `python recognize.py --mic`
- Preprocessing: resample to 16kHz mono, chunk into ~10s segments

### 2. Transcription Engine
- `mlx-whisper` with `tarteel-ai/whisper-base-ar-quran`
- Input: 16kHz mono audio array
- Output: Arabic text with diacritics

### 3. Text Normalizer
- Strip tashkeel (diacritics)
- Normalize alef variants (أ إ آ -> ا)
- Normalize taa marbuta (ة -> ه)
- Normalize alef maqsura (ى -> ي)

### 4. Verse Matcher
Two-phase approach:
1. **Surah narrowing**: N-gram overlap scoring to find top candidate surahs
2. **Ayah matching**: Levenshtein distance within candidates to find exact verse
3. **State tracking**: Bias toward sequential verses once position is established

### 5. Quran Text Database
- Source: Tanzil.net Simple-Clean + Uthmani
- Format: JSON with surah number, ayah number, Arabic text (both forms), surah name
- ~6,236 entries, ~2MB

## Data Sources
- **Model**: `tarteel-ai/whisper-base-ar-quran` from HuggingFace (~290MB)
- **Quran text**: Al Quran Cloud API (`api.alquran.cloud/v1/quran/quran-uthmani`) or Tanzil.net
- **Test audio**: EveryAyah.com verse-by-verse MP3s (URL: `everyayah.com/data/{Reciter}/{SSS}{AAA}.mp3`)

## Dependencies
- `mlx-whisper` - Apple MLX Whisper inference
- `soundfile` - audio file loading
- `numpy` - array operations
- `python-Levenshtein` - fast fuzzy string matching

## Success Criteria
- Given a ~10s audio clip of any reciter, correctly identify the surah:ayah
- Works fully offline after initial model download
- Runs on Apple Silicon Mac
