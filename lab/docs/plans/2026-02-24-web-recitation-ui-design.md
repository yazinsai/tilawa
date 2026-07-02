# Web Recitation UI — Design Doc

**Date:** 2026-02-24

## Goal

A web interface for continuous Quran recitation. The user clicks "Start", begins reciting, and sees the matched ayah with word-by-word highlighting in real time. Completed verses stack above as the recitation progresses.

## Architecture

```
Browser (React/Vite)                    Python Backend (FastAPI)
┌─────────────────────┐                ┌──────────────────────────┐
│ Mic → AudioWorklet  │                │  FastAPI + WebSocket     │
│   PCM chunks (16kHz)│───WebSocket───→│                          │
│                     │                │  Audio buffer            │
│ Verse display       │←──WebSocket────│    ↓                     │
│  - word highlighting│                │  mlx-whisper (3s chunks) │
│  - completed stack  │                │    ↓                     │
│  - confidence meter │                │  VersePositionTracker    │
└─────────────────────┘                │    ↓                     │
                                       │  JSON verse match state  │
                                       └──────────────────────────┘
```

## Backend — `web/server.py`

- FastAPI with one WebSocket endpoint (`/ws`)
- On connect: creates StreamingTranscriber + VersePositionTracker
- Receives binary 16kHz mono PCM from browser
- Buffers audio; runs mlx-whisper every 3 seconds of accumulated audio
- Feeds transcribed words into VersePositionTracker
- Sends JSON after each chunk:

```json
{
  "type": "verse_update",
  "current": {
    "surah": 2, "ayah": 255,
    "surah_name_en": "Al-Baqarah",
    "text_uthmani": "...",
    "words": ["اللَّهُ", "لَا", ...],
    "matched_indices": [0, 1, 2, 3],
    "word_position": 4,
    "total_words": 50,
    "confidence": 0.72
  },
  "completed": [
    {"surah": 1, "ayah": 1, "text_uthmani": "...", "surah_name_en": "Al-Fatiha"}
  ]
}
```

- Auto-advance: when verse reaches ~90% word coverage + confidence > 0.6, push to completed, reset tracker
- Serves built React app from `web/dist/` as static files
- Reuses existing: QuranDB, VersePositionTracker, normalize_arabic, StreamingTranscriber

## Frontend — `web/frontend/` (React + Vite + TypeScript)

Components:
- **App.tsx** — layout, WebSocket connection, state
- **CurrentVerse.tsx** — active verse with word-by-word highlighting (each word a `<span>`, matched words highlighted). RTL layout.
- **CompletedVerses.tsx** — scrollable stack of finished verses, dimmed
- **StartButton.tsx** — begins session, hides after start, mic begins streaming
- **StatusBar.tsx** — connection state, confidence, listening indicator

Audio capture: AudioWorklet resamples to 16kHz mono, sends PCM binary frames over WebSocket.

Style: Clean, large Arabic font (Amiri), dark background, generous spacing, centered.

## Decisions

- **3s chunk size** per streaming experiment results (best accuracy/latency balance)
- **mlx-whisper base** model for now (runs fast on Apple Silicon)
- **No overlap** between chunks (marginal benefit per benchmarks)
- **0.6 confidence threshold** for displaying matches
- **90% word coverage** threshold for auto-advancing to next verse
