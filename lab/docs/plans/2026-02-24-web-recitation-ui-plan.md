# Web Recitation UI — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Web interface that streams mic audio to a Python backend, shows the matched Quran ayah with word-by-word highlighting in real time, and stacks completed verses as the user recites continuously.

**Architecture:** Browser captures mic via AudioWorklet, streams 16kHz PCM over WebSocket to a FastAPI server. Server buffers audio, runs mlx-whisper on 3s chunks, feeds text through VersePositionTracker, sends verse match JSON back. React frontend renders word-highlighted verses.

**Tech Stack:** FastAPI + uvicorn (backend), mlx-whisper (ASR), React + Vite + TypeScript (frontend), WebSocket (transport)

---

### Task 1: Backend — WebSocket server with audio buffering

**Files:**
- Create: `web/server.py`
- Reuses: `src/offline_tarteel/quran_db.py`, `experiments/streaming-asr/verse_position_tracker.py`

**Step 1: Create `web/server.py`**

This is the complete backend. It:
- Accepts WebSocket connections at `/ws`
- Receives binary PCM audio frames from the browser
- Buffers audio until 3 seconds accumulate
- Runs mlx-whisper on each 3s chunk
- Feeds words into VersePositionTracker
- Sends verse match JSON back to the client
- Auto-advances when a verse reaches 90% coverage

```python
"""
WebSocket server for real-time Quran recitation tracking.

Receives 16kHz mono PCM audio from the browser, runs mlx-whisper
on 3-second chunks, matches against Quran verses, returns word-level
position tracking.
"""

import sys
import json
import asyncio
from pathlib import Path

import numpy as np
import mlx_whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "streaming-asr"))

from offline_tarteel.quran_db import QuranDB
from offline_tarteel.normalizer import normalize_arabic
from verse_position_tracker import VersePositionTracker

SAMPLE_RATE = 16000
CHUNK_SECONDS = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
MLX_MODEL = "mlx-community/whisper-base-mlx"
CONFIDENCE_DISPLAY_THRESHOLD = 0.4
ADVANCE_COVERAGE_THRESHOLD = 0.90
ADVANCE_CONFIDENCE_THRESHOLD = 0.6

app = FastAPI()

# Serve frontend build
DIST_DIR = Path(__file__).parent / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(DIST_DIR / "index.html")


def transcribe_chunk(audio: np.ndarray) -> str:
    """Run mlx-whisper on a single audio chunk."""
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MLX_MODEL,
        language="ar",
        condition_on_previous_text=False,
        hallucination_silence_threshold=1.0,
        no_speech_threshold=0.5,
    )
    return result.get("text", "").strip()


def is_hallucination(text: str) -> bool:
    """Detect hallucinated/repetitive Whisper output."""
    words = text.strip().split()
    if len(words) < 3:
        return False
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return True
    if len(words) > 30:
        return True
    return False


def is_silent(audio: np.ndarray, threshold: float = 0.005) -> bool:
    """Check if audio chunk is mostly silence."""
    return float(np.sqrt(np.mean(audio ** 2))) < threshold


def verse_match_to_dict(tracker: VersePositionTracker) -> dict | None:
    """Convert current tracker state to JSON-serializable dict."""
    m = tracker.current_match
    if not m:
        return None
    # Get uthmani words for the matched verse
    verse = tracker.db.get_verse(m.surah, m.ayah)
    uthmani_words = verse["text_uthmani"].split() if verse else m.verse_words
    return {
        "surah": m.surah,
        "ayah": m.ayah,
        "surah_name_en": m.surah_name_en,
        "text_uthmani": m.verse_text_uthmani,
        "words": uthmani_words,
        "matched_indices": m.matched_word_indices,
        "word_position": m.word_position,
        "total_words": m.total_words,
        "confidence": round(m.confidence, 3),
        "progress_pct": round(m.progress_pct, 1),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    db = QuranDB()
    tracker = VersePositionTracker(db)
    audio_buffer = np.array([], dtype=np.float32)
    completed: list[dict] = []

    try:
        while True:
            data = await ws.receive_bytes()

            # Browser sends Float32 PCM at 16kHz
            chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer = np.concatenate([audio_buffer, chunk])

            # Process when we have enough audio
            if len(audio_buffer) < CHUNK_SAMPLES:
                continue

            # Take 3s chunk
            process_chunk = audio_buffer[:CHUNK_SAMPLES]
            audio_buffer = audio_buffer[CHUNK_SAMPLES:]

            # Skip silence
            if is_silent(process_chunk):
                await ws.send_json({"type": "status", "status": "silence"})
                continue

            # Transcribe in a thread to not block the event loop
            text = await asyncio.to_thread(transcribe_chunk, process_chunk)

            if not text or is_hallucination(text):
                await ws.send_json({"type": "status", "status": "listening"})
                continue

            # Feed words into tracker
            words = text.split()
            match = tracker.update(words)

            current = None
            if match and match.confidence >= CONFIDENCE_DISPLAY_THRESHOLD:
                current = verse_match_to_dict(tracker)

                # Auto-advance if verse is nearly complete
                if (match.progress_pct >= ADVANCE_COVERAGE_THRESHOLD * 100
                        and match.confidence >= ADVANCE_CONFIDENCE_THRESHOLD):
                    completed.append({
                        "surah": match.surah,
                        "ayah": match.ayah,
                        "surah_name_en": match.surah_name_en,
                        "text_uthmani": match.verse_text_uthmani,
                    })
                    tracker.reset()
                    current = None

            await ws.send_json({
                "type": "verse_update",
                "current": current,
                "completed": completed,
            })

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
```

**Step 2: Install backend deps**

```bash
cd /Users/rock/ai/projects/offline-tarteel
source .venv/bin/activate
pip install fastapi uvicorn[standard]
```

**Step 3: Smoke-test server starts**

```bash
cd /Users/rock/ai/projects/offline-tarteel
source .venv/bin/activate
timeout 5 python web/server.py || true
```

Expected: Server starts, prints uvicorn startup message, then times out. No import errors.

**Step 4: Commit**

```bash
git add web/server.py
git commit -m "feat: add WebSocket server for real-time recitation tracking"
```

---

### Task 2: Frontend — Scaffold React app with Vite

**Files:**
- Create: `web/frontend/` (Vite scaffold)

**Step 1: Create Vite + React + TypeScript project**

```bash
cd /Users/rock/ai/projects/offline-tarteel/web
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

**Step 2: Add Amiri font**

```bash
cd /Users/rock/ai/projects/offline-tarteel/web/frontend
npm install @fontsource/amiri
```

**Step 3: Verify dev server starts**

```bash
cd /Users/rock/ai/projects/offline-tarteel/web/frontend
timeout 10 npm run dev || true
```

Expected: Vite dev server starts on port 5173.

**Step 4: Commit**

```bash
git add web/frontend/package.json web/frontend/package-lock.json web/frontend/tsconfig.json web/frontend/tsconfig.app.json web/frontend/tsconfig.node.json web/frontend/vite.config.ts web/frontend/index.html web/frontend/src/ web/frontend/public/
git commit -m "feat: scaffold React frontend with Vite"
```

---

### Task 3: Frontend — Audio capture with AudioWorklet

**Files:**
- Create: `web/frontend/public/audio-processor.js` (AudioWorklet processor)
- Create: `web/frontend/src/hooks/useAudioStream.ts`

**Step 1: Create AudioWorklet processor**

This runs in a separate audio thread. It captures raw audio from the mic, resamples to 16kHz, and posts Float32 PCM frames to the main thread.

```javascript
// audio-processor.js — AudioWorklet that captures mic audio
// Runs at the AudioContext's sample rate (usually 44100 or 48000)
// Downsamples to 16kHz and sends Float32 PCM chunks to main thread

class AudioStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._bufferSize = 4800; // send every 300ms at 16kHz
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0]; // mono
    const inputSampleRate = sampleRate; // global in AudioWorklet scope
    const outputSampleRate = 16000;
    const ratio = inputSampleRate / outputSampleRate;

    // Simple downsample by picking every Nth sample
    for (let i = 0; i < channelData.length; i += ratio) {
      this._buffer.push(channelData[Math.floor(i)]);
    }

    if (this._buffer.length >= this._bufferSize) {
      const chunk = new Float32Array(this._buffer);
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
      this._buffer = [];
    }

    return true;
  }
}

registerProcessor("audio-stream-processor", AudioStreamProcessor);
```

**Step 2: Create useAudioStream hook**

```typescript
// useAudioStream.ts — Hook that captures mic audio and streams to WebSocket

import { useRef, useCallback, useState } from "react";

type Status = "idle" | "connecting" | "streaming" | "error";

export function useAudioStream(wsUrl: string) {
  const [status, setStatus] = useState<Status>("idle");
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const start = useCallback(async () => {
    try {
      setStatus("connecting");

      // Connect WebSocket
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.onerror = () => reject(new Error("WebSocket failed"));
      });

      // Get mic
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      streamRef.current = stream;

      // Set up AudioWorklet
      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;

      await audioCtx.audioWorklet.addModule("/audio-processor.js");
      const source = audioCtx.createMediaStreamSource(stream);
      const processor = new AudioWorkletNode(audioCtx, "audio-stream-processor");

      processor.port.onmessage = (e: MessageEvent) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(e.data);
        }
      };

      source.connect(processor);
      // Don't connect to destination (we don't want to hear ourselves)

      setStatus("streaming");
    } catch (err) {
      console.error("Audio stream error:", err);
      setStatus("error");
    }
  }, [wsUrl]);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current?.close();
    wsRef.current?.close();
    wsRef.current = null;
    audioCtxRef.current = null;
    streamRef.current = null;
    setStatus("idle");
  }, []);

  return { status, start, stop, ws: wsRef };
}
```

**Step 3: Commit**

```bash
git add web/frontend/public/audio-processor.js web/frontend/src/hooks/useAudioStream.ts
git commit -m "feat: add AudioWorklet mic capture and WebSocket streaming hook"
```

---

### Task 4: Frontend — Verse display components

**Files:**
- Create: `web/frontend/src/types.ts`
- Create: `web/frontend/src/components/CurrentVerse.tsx`
- Create: `web/frontend/src/components/CompletedVerses.tsx`

**Step 1: Create shared types**

```typescript
// types.ts
export interface VerseState {
  surah: number;
  ayah: number;
  surah_name_en: string;
  text_uthmani: string;
  words: string[];
  matched_indices: number[];
  word_position: number;
  total_words: number;
  confidence: number;
  progress_pct: number;
}

export interface CompletedVerse {
  surah: number;
  ayah: number;
  surah_name_en: string;
  text_uthmani: string;
}

export interface ServerMessage {
  type: "verse_update" | "status";
  current?: VerseState | null;
  completed?: CompletedVerse[];
  status?: string;
}
```

**Step 2: Create CurrentVerse component**

Displays the active verse with word-by-word highlighting. Each word is a `<span>`. Matched words get a highlight class. RTL layout, large Arabic font.

```tsx
// CurrentVerse.tsx
import type { VerseState } from "../types";

interface Props {
  verse: VerseState;
}

export function CurrentVerse({ verse }: Props) {
  const matchedSet = new Set(verse.matched_indices);

  return (
    <div className="current-verse">
      <div className="verse-ref">
        {verse.surah_name_en} {verse.surah}:{verse.ayah}
        <span className="verse-confidence">
          {Math.round(verse.confidence * 100)}%
        </span>
      </div>
      <div className="verse-text" dir="rtl">
        {verse.words.map((word, i) => (
          <span
            key={i}
            className={`verse-word ${matchedSet.has(i) ? "matched" : ""}`}
          >
            {word}{" "}
          </span>
        ))}
      </div>
      <div className="verse-progress">
        <div
          className="verse-progress-bar"
          style={{ width: `${verse.progress_pct}%` }}
        />
      </div>
    </div>
  );
}
```

**Step 3: Create CompletedVerses component**

Scrollable stack of finished verses, dimmed, newest first.

```tsx
// CompletedVerses.tsx
import type { CompletedVerse } from "../types";

interface Props {
  verses: CompletedVerse[];
}

export function CompletedVerses({ verses }: Props) {
  if (verses.length === 0) return null;

  return (
    <div className="completed-verses">
      {[...verses].reverse().map((v, i) => (
        <div key={i} className="completed-verse">
          <div className="completed-ref">
            {v.surah_name_en} {v.surah}:{v.ayah}
          </div>
          <div className="completed-text" dir="rtl">
            {v.text_uthmani}
          </div>
        </div>
      ))}
    </div>
  );
}
```

**Step 4: Commit**

```bash
git add web/frontend/src/types.ts web/frontend/src/components/CurrentVerse.tsx web/frontend/src/components/CompletedVerses.tsx
git commit -m "feat: add verse display components with word highlighting"
```

---

### Task 5: Frontend — App shell and styling

**Files:**
- Modify: `web/frontend/src/App.tsx`
- Modify: `web/frontend/src/App.css` (replace entirely)
- Modify: `web/frontend/src/main.tsx`
- Modify: `web/frontend/src/index.css` (replace entirely)

**Step 1: Write App.tsx — ties everything together**

```tsx
// App.tsx
import { useState, useEffect, useCallback } from "react";
import { useAudioStream } from "./hooks/useAudioStream";
import { CurrentVerse } from "./components/CurrentVerse";
import { CompletedVerses } from "./components/CompletedVerses";
import type { VerseState, CompletedVerse, ServerMessage } from "./types";
import "@fontsource/amiri/400.css";
import "@fontsource/amiri/700.css";
import "./App.css";

const WS_URL = `ws://${window.location.hostname}:8765/ws`;

function App() {
  const { status, start, stop, ws } = useAudioStream(WS_URL);
  const [current, setCurrent] = useState<VerseState | null>(null);
  const [completed, setCompleted] = useState<CompletedVerse[]>([]);
  const [serverStatus, setServerStatus] = useState<string>("idle");

  const handleMessage = useCallback((event: MessageEvent) => {
    const msg: ServerMessage = JSON.parse(event.data);
    if (msg.type === "verse_update") {
      setCurrent(msg.current ?? null);
      if (msg.completed) setCompleted(msg.completed);
      setServerStatus("tracking");
    } else if (msg.type === "status") {
      setServerStatus(msg.status ?? "listening");
    }
  }, []);

  useEffect(() => {
    const socket = ws.current;
    if (!socket) return;
    socket.addEventListener("message", handleMessage);
    return () => socket.removeEventListener("message", handleMessage);
  }, [status, handleMessage, ws]);

  const handleStart = () => {
    start();
    setServerStatus("listening");
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Tarteel</h1>
        <div className="status-indicator">
          <span className={`status-dot ${status === "streaming" ? "active" : ""}`} />
          {status === "idle" && "Ready"}
          {status === "connecting" && "Connecting..."}
          {status === "streaming" && (serverStatus === "silence" ? "Silence" : serverStatus === "tracking" ? "Tracking" : "Listening...")}
          {status === "error" && "Error"}
        </div>
      </header>

      <main className="app-main">
        {status === "idle" ? (
          <button className="start-btn" onClick={handleStart}>
            Start Reciting
          </button>
        ) : (
          <>
            {current ? (
              <CurrentVerse verse={current} />
            ) : (
              status === "streaming" && (
                <div className="listening-prompt">Listening...</div>
              )
            )}
            <CompletedVerses verses={completed} />
            <button className="stop-btn" onClick={stop}>
              Stop
            </button>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
```

**Step 2: Write App.css**

```css
/* App.css */
.app {
  max-width: 720px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 3rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid #2a2a2a;
}

.app-header h1 {
  font-size: 1.25rem;
  font-weight: 400;
  color: #888;
  letter-spacing: 0.05em;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  color: #666;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #444;
  transition: background 0.3s;
}

.status-dot.active {
  background: #4ade80;
  box-shadow: 0 0 8px #4ade8066;
}

.app-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
}

/* Start / Stop buttons */
.start-btn {
  margin-top: 30vh;
  padding: 1rem 3rem;
  font-size: 1.25rem;
  font-family: inherit;
  background: #1a1a1a;
  color: #e0e0e0;
  border: 1px solid #333;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;
}

.start-btn:hover {
  background: #222;
  border-color: #555;
}

.stop-btn {
  margin-top: 2rem;
  padding: 0.5rem 1.5rem;
  font-size: 0.85rem;
  font-family: inherit;
  background: transparent;
  color: #666;
  border: 1px solid #333;
  border-radius: 8px;
  cursor: pointer;
}

.stop-btn:hover {
  color: #e74c3c;
  border-color: #e74c3c;
}

/* Listening state */
.listening-prompt {
  margin-top: 20vh;
  font-size: 1.5rem;
  color: #555;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}

/* Current verse */
.current-verse {
  width: 100%;
  margin-bottom: 2rem;
}

.verse-ref {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.85rem;
  color: #888;
  margin-bottom: 1rem;
}

.verse-confidence {
  font-size: 0.75rem;
  color: #666;
  background: #1a1a1a;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
}

.verse-text {
  font-family: "Amiri", serif;
  font-size: 2rem;
  line-height: 2.2;
  color: #555;
  text-align: right;
}

.verse-word {
  transition: color 0.3s, text-shadow 0.3s;
}

.verse-word.matched {
  color: #e0e0e0;
  text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
}

.verse-progress {
  margin-top: 1rem;
  height: 2px;
  background: #1a1a1a;
  border-radius: 1px;
  overflow: hidden;
}

.verse-progress-bar {
  height: 100%;
  background: #4ade80;
  transition: width 0.5s ease;
  border-radius: 1px;
}

/* Completed verses */
.completed-verses {
  width: 100%;
  margin-top: 2rem;
  padding-top: 2rem;
  border-top: 1px solid #1a1a1a;
}

.completed-verse {
  margin-bottom: 1.5rem;
  opacity: 0.4;
}

.completed-ref {
  font-size: 0.75rem;
  color: #555;
  margin-bottom: 0.5rem;
}

.completed-text {
  font-family: "Amiri", serif;
  font-size: 1.25rem;
  line-height: 1.8;
  color: #666;
  text-align: right;
  direction: rtl;
}
```

**Step 3: Write index.css (global reset)**

```css
/* index.css */
*,
*::before,
*::after {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #0a0a0a;
  color: #e0e0e0;
  -webkit-font-smoothing: antialiased;
}
```

**Step 4: Update main.tsx (remove StrictMode double-render for audio)**

```tsx
// main.tsx
import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

createRoot(document.getElementById("root")!).render(<App />);
```

**Step 5: Verify frontend builds**

```bash
cd /Users/rock/ai/projects/offline-tarteel/web/frontend
npm run build
```

Expected: Build succeeds, `dist/` directory created.

**Step 6: Commit**

```bash
git add web/frontend/src/App.tsx web/frontend/src/App.css web/frontend/src/index.css web/frontend/src/main.tsx
git commit -m "feat: wire up App shell with audio streaming and verse display"
```

---

### Task 6: Integration test — end-to-end recitation

**Step 1: Build frontend and start server**

```bash
cd /Users/rock/ai/projects/offline-tarteel/web/frontend
npm run build
cd /Users/rock/ai/projects/offline-tarteel
source .venv/bin/activate
python web/server.py
```

**Step 2: Test in browser**

Open `http://localhost:8765` in Chrome/Safari. Click "Start Reciting". Recite Bismillah or Al-Fatiha. Verify:
- Green dot appears (streaming status)
- After ~3 seconds, verse text appears with word highlighting
- Words light up as more audio is processed
- When verse completes, it moves to completed stack dimmed

**Step 3: Fix any issues found during testing**

**Step 4: Final commit**

```bash
git add -A  # only if no sensitive files
git commit -m "feat: complete web recitation UI with real-time verse tracking"
```

---

### Task 7: Vite proxy for development

**Files:**
- Modify: `web/frontend/vite.config.ts`

**Step 1: Add WebSocket proxy**

So `npm run dev` can proxy WebSocket to the Python backend:

```typescript
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ws": {
        target: "ws://localhost:8765",
        ws: true,
      },
    },
  },
});
```

**Step 2: Update App.tsx WS_URL for dev**

Change the WS_URL to use the current page's host (works for both dev proxy and production):

```typescript
const WS_URL = `ws://${window.location.host}/ws`;
```

**Step 3: Commit**

```bash
git add web/frontend/vite.config.ts web/frontend/src/App.tsx
git commit -m "feat: add Vite dev proxy for WebSocket"
```
