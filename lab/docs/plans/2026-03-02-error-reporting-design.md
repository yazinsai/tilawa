# Error Reporting & Training Data Collection — Design

## Goal

Allow users to report recognition errors by submitting an audio clip of their recitation along with the correct surah/ayah. Collected data enables model fine-tuning.

## UX Changes

### Start/Stop Recording (replaces auto-listen)

- **Start button**: Opens mic, begins buffering audio and feeding recognition pipeline.
- **Stop button**: Ends recording, retains full audio buffer in memory.
- After stopping, user sees recognized verse(s) plus two actions:
  - **Report Error** — opens the report dialog
  - **Start Again** — begins a new session

Sessions are discrete; audio buffers are bounded to a single session.

### Report Error Dialog

1. **Audio playback** — waveform/playback bar for the recorded clip
2. **Verse selector** — two dropdowns (surah 1-114, ayah 1-N) pre-filled with model's prediction
3. **Model prediction** — read-only display of what the model recognized
4. **Notes** — optional textarea for free-text description
5. **Submit** — sends audio WAV + metadata to backend

## Backend (Hono)

Lightweight TypeScript server replaces nginx. Single process serves static frontend + API.

### Routes

| Method | Path | Description |
|--------|------|-------------|
| `*` | `/` | Serves Vite static build from `/dist` |
| `POST` | `/api/reports` | Accepts multipart form (WAV + JSON metadata) |
| `GET` | `/admin` | Password-protected admin page |
| `GET` | `/api/reports` | Lists all reports (admin) |
| `GET` | `/api/reports/:id/audio` | Streams audio file for playback |

### Storage

Dokku persistent volume mounted at `/storage`:

```
/storage/reports/
  {uuid}/
    audio.wav
    meta.json
```

`meta.json` schema:
```json
{
  "id": "uuid",
  "surah": 2,
  "ayah": 255,
  "modelPrediction": { "surah": 2, "ayah": 254, "confidence": 0.62 },
  "notes": "Model confused ayat al-kursi ending",
  "timestamp": "2026-03-02T12:00:00Z",
  "userAgent": "Mozilla/5.0 ...",
  "audioDurationMs": 4500
}
```

### Auth

Admin page gated by single shared password via `ADMIN_PASSWORD` env var.

## Admin Page

Server-rendered HTML at `/admin`:
- Password gate (cookie-based session after auth)
- Table: timestamp, expected verse, model prediction, notes preview
- Row expand: full notes, audio player, all metadata
- No bulk export (SSH + tar for now)

## Audio Format

16kHz mono WAV — same as recognition pipeline input. AudioWorklet already resamples to 16kHz; we accumulate chunks in a Float32Array buffer and encode to WAV on submit.

## Docker Changes

Replace nginx final stage with Node.js:

```dockerfile
FROM node:22-slim
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/server ./server
CMD ["node", "server/index.js"]
```

Dokku: `dokku storage:mount app-name /var/lib/dokku/data/storage/app-name:/app/storage`

## Decisions

- **Hono over Express/FastAPI**: Same language as frontend, minimal deps, single process
- **Disk storage over DB**: Simple, no external deps, audio files are the primary artifact
- **Start/Stop replaces auto-listen**: Bounded sessions, cleaner UX, manageable audio buffers
- **Pre-filled dropdowns**: Reduces friction; user corrects from model's guess rather than starting from scratch
