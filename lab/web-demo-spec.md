# Quran Verse Recognition Web Demo — Spec

## Overview

A lightweight web app that streams mic audio and shows recognized Quran verses in real-time using mushaf-style Arabic text. Python backend (FastAPI + WebSocket) runs the FastConformer model locally. Vite frontend handles mic capture and verse rendering. No cloud dependency — fully offline (local).

## Architecture

```
Browser (Vite + TS)                    Python Backend (FastAPI)
┌─────────────────────┐                ┌──────────────────────────┐
│  WebAudio API       │   WebSocket    │  FastConformer (NeMo)    │
│  Mic capture        │ ──── PCM ───→  │  ASR transcription       │
│  16kHz resampling   │                │                          │
│                     │  ← JSON ─────  │  QuranDB fuzzy matching  │
│  Verse renderer     │                │  Span-aware (1-4 verses) │
│  Arabic font        │                └──────────────────────────┘
│  Pulse animation    │
└─────────────────────┘
```

- **Backend**: FastAPI with WebSocket endpoint. Runs the existing FastConformer pipeline + QuranDB matching. Reuses all existing Python code.
- **Frontend**: Vite + vanilla TypeScript (lightweight, no framework). Handles mic, WebSocket streaming, and verse rendering.
- **Model**: `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` — 85% SeqAcc, 0.33s latency, 115 MB. No ONNX conversion needed since it runs server-side via NeMo.
- **Matching**: Existing `shared/quran_db.py` with Levenshtein fuzzy matching + multi-verse span support.

## Audio Pipeline

1. **Mic capture**: `navigator.mediaDevices.getUserMedia()` → WebAudio API
2. **Resampling**: Downsample to 16kHz mono PCM (FastConformer's expected input)
3. **Streaming**: Send audio chunks over WebSocket every ~1-2 seconds (continuous, overlapping window)
4. **Processing**: Backend accumulates audio, runs FastConformer ASR, matches against QuranDB
5. **Response**: Backend sends JSON with recognized verse(s) or raw transcript

### WebSocket Message Format

**Client → Server** (binary): Raw PCM float32 audio chunks at 16kHz

**Server → Client** (JSON):
```json
{
  "type": "verse_match",
  "surah": 2,
  "ayah": 255,
  "verse_text": "ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ ٱلْحَىُّ ٱلْقَيُّومُ...",
  "surah_name": "البقرة",
  "confidence": 0.92,
  "surrounding_verses": [...]
}
```
```json
{
  "type": "raw_transcript",
  "text": "الله لا اله الا هو",
  "confidence": 0.45
}
```

## Recognition Flow

- **Continuous streaming**: Audio is sent to the backend every ~1-2 seconds
- **Silence detection** (optional optimization): Backend can skip processing silent chunks
- **Overlapping window**: Keep a rolling buffer of the last ~5-10 seconds for context
- **Result deduplication**: Don't re-display a verse that's already shown unless position changed

## UI / UX

### Empty State
- Blank canvas — completely empty page
- Subtle pulse/glow animation at the bottom center indicating the app is listening
- No text, no instructions. Verses appear from nothing when recitation is detected.

### Always Listening
- Mic starts automatically on page load (after browser permission grant)
- No start/stop button. The app is always on.
- If mic permission is denied, show a single line requesting permission.

### Verse Display
- **Font**: Quranic Naskh font (KFGQ Hafs or Amiri — whichever renders best at web sizes)
- **Tashkeel**: Full diacritical marks on all verse text
- **Layout**: Centered narrow column (~600px max), RTL text direction
- **Dynamic expansion**: First recognized verse appears alone. As consecutive verses are recognized, surrounding context expands — showing more of the surah around the current position.
- **Stacking**: New verse groups stack below previous ones as the user moves to different parts of the Quran.
- **Active verse highlight**: The currently recognized verse is visually distinct (slightly bolder, or subtle background), surrounding context verses are lighter.

### Raw Transcript Fallback
- When the model hears something but can't confidently match a verse, show the raw ASR transcript
- Displayed in a lighter/grayed style below the listening indicator
- Replaces itself with the next recognition attempt (doesn't stack)

### Verse Metadata
- Surah name and ayah number are hidden by default
- Revealed on hover (desktop) or tap (mobile)
- Displayed as "سورة البقرة — آية ٢٥٥" in small text above/below the verse

### Listening Indicator
- Subtle pulse/glow animation (breathing effect)
- Positioned at the bottom center of the page
- Small, unobtrusive — doesn't compete with the Quran text
- Changes color/intensity slightly when audio is detected vs silence

### Design
- **Aesthetic**: Minimal & clean. White/cream background. Lots of whitespace.
- **Typography**: Arabic text is the hero. Large, beautiful Quranic font.
- **Colors**: Neutral palette. Warm off-white background, dark text. Active verse gets a very subtle warm highlight.
- **Responsive**: Centered column on desktop, full-width with padding on mobile. Same experience, adapts naturally.
- **No dark mode** (keep it simple for the demo).

## Technical Details

### Frontend (Vite + TypeScript)
- `vite` + vanilla TypeScript (no React/Vue)
- WebAudio API for mic capture + resampling
- WebSocket client for streaming
- CSS-only animations (pulse/glow)
- Self-hosted Quranic web font
- No build-time dependencies beyond Vite

### Backend (Python FastAPI)
- FastAPI with `websockets` support
- NeMo FastConformer model loaded on startup
- Existing `shared/quran_db.py` for verse matching
- Audio accumulation buffer with sliding window
- Runs on `localhost:8000`

### Startup Flow
1. User runs `python server.py` (or `uvicorn`)
2. Model loads (~5-10 seconds)
3. User opens `localhost:5173` (Vite dev) or built static files served by FastAPI
4. Browser requests mic permission
5. Audio streaming begins
6. Verses appear as recognized

### No Persistence
- Fresh slate on every page load
- No localStorage, no session history
- Stateless — each visit is a new experience

### No PWA
- No service worker, no manifest, no install prompt
- Just a website that works when the local server is running

## Out of Scope
- Manual search/navigation to specific verses
- Dark mode
- Translation display
- Verse audio playback
- User accounts or history
- Deployment to a remote server
- ONNX/WASM browser-only mode (future consideration)
