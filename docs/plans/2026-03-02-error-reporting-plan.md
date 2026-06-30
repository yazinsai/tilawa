# Error Reporting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace auto-listen with Start/Stop recording, add error report submission with audio + metadata, add a Hono backend for storage, and a simple admin page.

**Architecture:** Hono server replaces nginx — serves static Vite build + API routes. Audio buffered in main thread during Start/Stop sessions. Reports saved as WAV + JSON to Dokku persistent storage. Admin page is server-rendered HTML behind a password.

**Tech Stack:** Hono (server), Vite (frontend build), vanilla TS (frontend), WAV encoding (browser), Dokku persistent storage (disk).

---

### Task 1: Add Hono server scaffolding

**Files:**
- Create: `web/frontend/server/index.ts`
- Create: `web/frontend/server/tsconfig.json`
- Modify: `web/frontend/package.json`

**Step 1: Install dependencies**

Run: `cd /Users/rock/ai/projects/offline-tarteel/web/frontend && npm install hono @hono/node-server`

**Step 2: Create server tsconfig**

Create `web/frontend/server/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "outDir": "../dist-server",
    "rootDir": ".",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "declaration": false
  },
  "include": ["./**/*.ts"]
}
```

**Step 3: Create minimal Hono server**

Create `web/frontend/server/index.ts`:
```typescript
import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";

const app = new Hono();

// API routes will go here
app.get("/api/health", (c) => c.json({ ok: true }));

// Serve static files from Vite build
app.use("/*", serveStatic({ root: "./dist" }));

// SPA fallback
app.get("/*", serveStatic({ root: "./dist", path: "index.html" }));

const port = parseInt(process.env.PORT || "5000");
console.log(`Server running on port ${port}`);
serve({ fetch: app.fetch, port });
```

**Step 4: Add build scripts to package.json**

Add to `web/frontend/package.json` scripts:
```json
"build:server": "npx esbuild server/index.ts --bundle --platform=node --format=esm --outfile=dist-server/index.mjs --packages=external",
"start": "node dist-server/index.mjs"
```

Also add `esbuild` as a devDependency:
Run: `cd /Users/rock/ai/projects/offline-tarteel/web/frontend && npm install -D esbuild`

**Step 5: Test the server locally**

Run: `cd /Users/rock/ai/projects/offline-tarteel/web/frontend && npm run build && npm run build:server && npm run start`
Expected: Server starts on port 5000, `curl http://localhost:5000/api/health` returns `{"ok":true}`

**Step 6: Commit**

```bash
git add web/frontend/server/index.ts web/frontend/server/tsconfig.json web/frontend/package.json web/frontend/package-lock.json
git commit -m "feat: add Hono server scaffolding"
```

---

### Task 2: Update Dockerfile to use Hono instead of nginx

**Files:**
- Modify: `web/frontend/Dockerfile`

**Step 1: Rewrite the Dockerfile**

Replace `web/frontend/Dockerfile` with:
```dockerfile
# Stage 1: Build frontend
FROM node:22-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build
RUN npm run build:server

# Stage 2: Runtime
FROM node:22-slim
WORKDIR /app

# Copy built frontend
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/dist-server ./dist-server

# Copy package files for production deps only
COPY --from=builder /app/package.json /app/package-lock.json ./
RUN npm ci --omit=dev

# Download current Cyberistic ONNX model
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -L -o dist/fastconformer_full_mixed.onnx \
    https://github.com/yazinsai/offline-tarteel/releases/download/v0.2.0/fastconformer_full_mixed.onnx

# Create storage directory
RUN mkdir -p /app/storage/reports

EXPOSE 5000
CMD ["node", "dist-server/index.mjs"]
```

**Step 2: Commit**

```bash
git add web/frontend/Dockerfile
git commit -m "feat: switch Dockerfile from nginx to Hono node server"
```

---

### Task 3: Add report submission API endpoint

**Files:**
- Create: `web/frontend/server/reports.ts`
- Modify: `web/frontend/server/index.ts`

**Step 1: Create reports route module**

Create `web/frontend/server/reports.ts`:
```typescript
import { Hono } from "hono";
import { randomUUID } from "node:crypto";
import { mkdir, writeFile, readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const STORAGE_DIR = process.env.STORAGE_DIR || "./storage/reports";

export const reportsApp = new Hono();

// POST /api/reports — accept audio + metadata
reportsApp.post("/", async (c) => {
  const form = await c.req.formData();
  const audio = form.get("audio") as File | null;
  const metaRaw = form.get("metadata") as string | null;

  if (!audio || !metaRaw) {
    return c.json({ error: "Missing audio or metadata" }, 400);
  }

  let meta: Record<string, unknown>;
  try {
    meta = JSON.parse(metaRaw);
  } catch {
    return c.json({ error: "Invalid metadata JSON" }, 400);
  }

  const id = randomUUID();
  const dir = join(STORAGE_DIR, id);
  await mkdir(dir, { recursive: true });

  // Save audio
  const audioBuffer = Buffer.from(await audio.arrayBuffer());
  await writeFile(join(dir, "audio.wav"), audioBuffer);

  // Save metadata
  const fullMeta = {
    id,
    ...meta,
    timestamp: new Date().toISOString(),
    userAgent: c.req.header("user-agent") || "",
    audioSizeBytes: audioBuffer.length,
  };
  await writeFile(join(dir, "meta.json"), JSON.stringify(fullMeta, null, 2));

  return c.json({ id, status: "saved" }, 201);
});

// GET /api/reports — list all reports
reportsApp.get("/", async (c) => {
  try {
    const entries = await readdir(STORAGE_DIR);
    const reports = [];
    for (const entry of entries) {
      try {
        const metaPath = join(STORAGE_DIR, entry, "meta.json");
        const raw = await readFile(metaPath, "utf-8");
        reports.push(JSON.parse(raw));
      } catch {
        // Skip broken entries
      }
    }
    // Sort newest first
    reports.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    return c.json(reports);
  } catch {
    return c.json([]);
  }
});

// GET /api/reports/:id/audio — stream audio file
reportsApp.get("/:id/audio", async (c) => {
  const id = c.req.param("id");
  const filePath = join(STORAGE_DIR, id, "audio.wav");
  try {
    const data = await readFile(filePath);
    return new Response(data, {
      headers: {
        "Content-Type": "audio/wav",
        "Content-Length": String(data.length),
      },
    });
  } catch {
    return c.json({ error: "Not found" }, 404);
  }
});
```

**Step 2: Wire reports into main server**

Add to `web/frontend/server/index.ts`, before the static file middleware:
```typescript
import { reportsApp } from "./reports";

// Mount reports API
app.route("/api/reports", reportsApp);
```

**Step 3: Test the endpoint**

Run the server and test with curl:
```bash
# Create a test WAV file (1 second of silence)
printf 'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00' > /tmp/test.wav

curl -X POST http://localhost:5000/api/reports \
  -F "audio=@/tmp/test.wav" \
  -F 'metadata={"surah":2,"ayah":255,"modelPrediction":{"surah":2,"ayah":254,"confidence":0.62},"notes":"test"}'
```
Expected: `{"id":"<uuid>","status":"saved"}` with 201 status.

Then test listing:
```bash
curl http://localhost:5000/api/reports
```
Expected: Array containing the report just created.

**Step 4: Commit**

```bash
git add web/frontend/server/reports.ts web/frontend/server/index.ts
git commit -m "feat: add report submission and listing API endpoints"
```

---

### Task 4: Replace auto-listen with Start/Stop recording in UI

**Files:**
- Modify: `web/frontend/index.html`
- Modify: `web/frontend/src/main.ts`
- Modify: `web/frontend/src/style.css`

**Step 1: Update index.html**

Add recording controls to the `#app` div. Replace the `listening-indicator` and `listening-status` sections with new Start/Stop controls and a post-recording action bar:

After the `#loading-status` div, add:
```html
<div id="ready-state" hidden>
  <p class="ready-prompt">Tap to begin recitation</p>
  <button id="btn-start" class="rec-btn rec-btn--start">Start</button>
</div>
<div id="recording-state" hidden>
  <button id="btn-stop" class="rec-btn rec-btn--stop">Stop</button>
</div>
<div id="post-recording" hidden>
  <div class="post-actions">
    <button id="btn-report" class="action-btn action-btn--report">Report Error</button>
    <button id="btn-restart" class="action-btn action-btn--restart">Start Again</button>
  </div>
</div>
```

Remove or hide the old `#listening-status` and `#listening-indicator` sections (keep them in HTML but we'll stop using them).

**Step 2: Rewrite audio flow in main.ts**

The key changes:
1. Remove auto-call to `startAudio()` from DOMContentLoaded
2. `startAudio()` opens mic + starts streaming to worker + starts accumulating a session audio buffer
3. `stopAudio()` stops mic, retains the session buffer, shows post-recording actions
4. Add a `sessionAudioBuffer: Float32Array` to state that accumulates all 16kHz chunks during a session
5. Wire up Start/Stop/Report/Restart button click handlers

State additions:
```typescript
// Add to state object:
sessionAudioChunks: [] as Float32Array[],
lastModelPrediction: null as { surah: number; ayah: number; confidence: number } | null,
```

In the audio worklet message handler, also push each chunk to `state.sessionAudioChunks`:
```typescript
processor.port.onmessage = (e: MessageEvent) => {
  const samples = new Float32Array(e.data as ArrayBuffer);
  // Save to session buffer
  state.sessionAudioChunks.push(samples.slice());
  // Send to worker for recognition
  if (state.worker) {
    state.worker.postMessage({ type: "audio", samples }, [samples.buffer]);
  }
};
```

On `verse_match` messages, update `state.lastModelPrediction`:
```typescript
state.lastModelPrediction = { surah: msg.surah, ayah: msg.ayah, confidence: msg.confidence };
```

Button handlers:
- **Start**: calls `startAudio()`, shows `#recording-state`, hides `#ready-state`
- **Stop**: stops mic stream tracks, shows `#post-recording` with verse results, hides `#recording-state`
- **Start Again**: clears session buffer and prediction, shows `#ready-state`, hides `#post-recording`, clears verses
- **Report Error**: opens the error report dialog (Task 5)

`stopAudio()`:
```typescript
function stopAudio(): void {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  if (state.audioCtx) {
    state.audioCtx.close();
    state.audioCtx = null;
  }
  state.isActive = false;
}
```

**Step 3: Add CSS for new buttons**

Add styles for `.rec-btn`, `.action-btn`, `#ready-state`, `#recording-state`, `#post-recording` to `style.css`. Keep the warm, minimal aesthetic:

```css
/* ── Recording controls ── */
#ready-state, #recording-state, #post-recording {
  text-align: center;
  padding: 2rem 1rem;
  animation: fade-in 0.4s ease-out;
}

.ready-prompt {
  font-family: var(--font-system);
  font-size: 1.1rem;
  color: var(--text-muted);
  margin-bottom: 1.5rem;
}

.rec-btn {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  font-family: var(--font-system);
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.rec-btn:active {
  transform: scale(0.95);
}

.rec-btn--start {
  background: #b8986a;
  color: #fff;
  box-shadow: 0 2px 12px rgba(184, 152, 106, 0.3);
}

.rec-btn--stop {
  background: #c0564b;
  color: #fff;
  box-shadow: 0 2px 12px rgba(192, 86, 75, 0.3);
}

.post-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}

.action-btn {
  padding: 0.6rem 1.5rem;
  border-radius: 8px;
  border: 1px solid var(--accent-warm-strong);
  font-family: var(--font-system);
  font-size: 0.9rem;
  cursor: pointer;
  transition: background 0.2s ease;
}

.action-btn--report {
  background: var(--accent-warm);
  color: var(--text);
}

.action-btn--report:hover {
  background: var(--accent-warm-strong);
}

.action-btn--restart {
  background: transparent;
  color: var(--text-light);
  border-color: rgba(200, 190, 170, 0.3);
}

.action-btn--restart:hover {
  background: var(--accent-warm);
}
```

**Step 4: Test manually in browser**

Run: `cd /Users/rock/ai/projects/offline-tarteel/web/frontend && npm run dev`
Open `http://localhost:5173`, verify:
- Model loads, then Start button appears
- Click Start → mic activates, recognition runs
- Click Stop → mic stops, post-recording buttons appear
- Click Start Again → back to ready state

**Step 5: Commit**

```bash
git add web/frontend/index.html web/frontend/src/main.ts web/frontend/src/style.css
git commit -m "feat: replace auto-listen with Start/Stop recording controls"
```

---

### Task 5: Add WAV encoder utility

**Files:**
- Create: `web/frontend/src/lib/wav-encoder.ts`

**Step 1: Create WAV encoder**

Create `web/frontend/src/lib/wav-encoder.ts`:
```typescript
/**
 * Encode Float32Array audio samples to WAV format (16-bit PCM).
 */
export function encodeWav(samples: Float32Array, sampleRate = 16000): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const dataLength = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeString(view, 8, "WAVE");

  // fmt chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  writeString(view, 36, "data");
  view.setUint32(40, dataLength, true);

  // Write PCM samples (clamp to [-1, 1] and convert to 16-bit)
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

/**
 * Concatenate an array of Float32Arrays into a single Float32Array.
 */
export function concatChunks(chunks: Float32Array[]): Float32Array {
  let totalLength = 0;
  for (const chunk of chunks) totalLength += chunk.length;
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}
```

**Step 2: Commit**

```bash
git add web/frontend/src/lib/wav-encoder.ts
git commit -m "feat: add WAV encoder utility for audio export"
```

---

### Task 6: Add error report dialog UI

**Files:**
- Modify: `web/frontend/index.html`
- Create: `web/frontend/src/report-dialog.ts`
- Modify: `web/frontend/src/main.ts`
- Modify: `web/frontend/src/style.css`

**Step 1: Add dialog HTML to index.html**

Add before closing `</div>` of `#app`:
```html
<dialog id="report-dialog">
  <div class="dialog-content">
    <h2 class="dialog-title">Report Error</h2>

    <div class="dialog-section">
      <label class="dialog-label">Your recording</label>
      <audio id="report-audio" controls></audio>
    </div>

    <div class="dialog-section">
      <label class="dialog-label">What were you reciting?</label>
      <div class="verse-selectors">
        <select id="report-surah"></select>
        <select id="report-ayah"></select>
      </div>
    </div>

    <div class="dialog-section">
      <label class="dialog-label">Model predicted</label>
      <p id="report-prediction" class="prediction-text"></p>
    </div>

    <div class="dialog-section">
      <label class="dialog-label">Notes (optional)</label>
      <textarea id="report-notes" rows="3" placeholder="Describe the issue..."></textarea>
    </div>

    <div class="dialog-actions">
      <button id="btn-submit-report" class="action-btn action-btn--report">Submit</button>
      <button id="btn-cancel-report" class="action-btn action-btn--restart">Cancel</button>
    </div>

    <p id="report-status" class="report-status" hidden></p>
  </div>
</dialog>
```

**Step 2: Create report-dialog.ts**

Create `web/frontend/src/report-dialog.ts` that handles:
1. Populating the surah dropdown (1-114 with Arabic + English names from quran.json)
2. Populating the ayah dropdown based on selected surah
3. Pre-filling with model's prediction
4. Playing back recorded audio via an object URL
5. Submitting the form data to `/api/reports`

```typescript
import { encodeWav, concatChunks } from "./lib/wav-encoder";
import type { QuranVerse } from "./lib/types";

interface ReportDialogOptions {
  audioChunks: Float32Array[];
  modelPrediction: { surah: number; ayah: number; confidence: number } | null;
  quranData: QuranVerse[];
}

const $dialog = document.getElementById("report-dialog") as HTMLDialogElement;
const $surah = document.getElementById("report-surah") as HTMLSelectElement;
const $ayah = document.getElementById("report-ayah") as HTMLSelectElement;
const $audio = document.getElementById("report-audio") as HTMLAudioElement;
const $prediction = document.getElementById("report-prediction")!;
const $notes = document.getElementById("report-notes") as HTMLTextAreaElement;
const $submit = document.getElementById("btn-submit-report")!;
const $cancel = document.getElementById("btn-cancel-report")!;
const $status = document.getElementById("report-status")!;

let currentAudioBlob: Blob | null = null;
let currentQuranData: QuranVerse[] = [];

// Build surah list (called once after quran.json loads)
export function initSurahDropdown(quranData: QuranVerse[]): void {
  currentQuranData = quranData;
  const surahs = new Map<number, { name: string; nameEn: string }>();
  for (const v of quranData) {
    if (!surahs.has(v.surah)) {
      surahs.set(v.surah, { name: v.surah_name, nameEn: v.surah_name_en });
    }
  }
  $surah.innerHTML = "";
  for (const [num, info] of surahs) {
    const opt = document.createElement("option");
    opt.value = String(num);
    opt.textContent = `${num}. ${info.nameEn} — ${info.name}`;
    $surah.appendChild(opt);
  }
  $surah.addEventListener("change", () => updateAyahDropdown(parseInt($surah.value)));
}

function updateAyahDropdown(surahNum: number): void {
  const verses = currentQuranData.filter(v => v.surah === surahNum);
  $ayah.innerHTML = "";
  for (const v of verses) {
    const opt = document.createElement("option");
    opt.value = String(v.ayah);
    opt.textContent = `Ayah ${v.ayah}`;
    $ayah.appendChild(opt);
  }
}

export function openReportDialog(opts: ReportDialogOptions): void {
  // Encode audio
  const combined = concatChunks(opts.audioChunks);
  currentAudioBlob = encodeWav(combined);
  const url = URL.createObjectURL(currentAudioBlob);
  $audio.src = url;

  // Pre-fill with model prediction
  if (opts.modelPrediction) {
    $surah.value = String(opts.modelPrediction.surah);
    updateAyahDropdown(opts.modelPrediction.surah);
    $ayah.value = String(opts.modelPrediction.ayah);
    const pred = opts.modelPrediction;
    $prediction.textContent = `Surah ${pred.surah}, Ayah ${pred.ayah} (${Math.round(pred.confidence * 100)}% confidence)`;
  } else {
    $surah.value = "1";
    updateAyahDropdown(1);
    $prediction.textContent = "No prediction available";
  }

  $notes.value = "";
  $status.hidden = true;
  $submit.removeAttribute("disabled");
  $dialog.showModal();
}

// Submit handler
$submit.addEventListener("click", async () => {
  if (!currentAudioBlob) return;
  $submit.setAttribute("disabled", "true");
  $status.textContent = "Submitting...";
  $status.hidden = false;

  const metadata = {
    surah: parseInt($surah.value),
    ayah: parseInt($ayah.value),
    modelPrediction: $prediction.textContent,
    notes: $notes.value.trim(),
  };

  const formData = new FormData();
  formData.append("audio", currentAudioBlob, "recording.wav");
  formData.append("metadata", JSON.stringify(metadata));

  try {
    const res = await fetch("/api/reports", { method: "POST", body: formData });
    if (res.ok) {
      $status.textContent = "Report submitted. Thank you!";
      setTimeout(() => $dialog.close(), 1500);
    } else {
      const err = await res.json();
      $status.textContent = `Error: ${err.error || "Unknown error"}`;
      $submit.removeAttribute("disabled");
    }
  } catch (e) {
    $status.textContent = "Network error. Please try again.";
    $submit.removeAttribute("disabled");
  }
});

// Cancel handler
$cancel.addEventListener("click", () => {
  $dialog.close();
  if ($audio.src) URL.revokeObjectURL($audio.src);
});
```

**Step 3: Wire into main.ts**

In `main.ts`:
1. Import `initSurahDropdown` and `openReportDialog` from `./report-dialog`
2. After quran data loads, call `initSurahDropdown(state.quranData!)`
3. In the Report Error button handler:
```typescript
document.getElementById("btn-report")!.addEventListener("click", () => {
  openReportDialog({
    audioChunks: state.sessionAudioChunks,
    modelPrediction: state.lastModelPrediction,
    quranData: state.quranData!,
  });
});
```

**Step 4: Add dialog CSS**

Add to `style.css`:
```css
/* ── Report dialog ── */
#report-dialog {
  border: none;
  border-radius: 12px;
  padding: 0;
  max-width: 480px;
  width: calc(100% - 2rem);
  background: var(--bg);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
}

#report-dialog::backdrop {
  background: rgba(44, 36, 22, 0.4);
  backdrop-filter: blur(4px);
}

.dialog-content {
  padding: 1.5rem;
}

.dialog-title {
  font-family: var(--font-system);
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 1.25rem;
}

.dialog-section {
  margin-bottom: 1rem;
}

.dialog-label {
  display: block;
  font-family: var(--font-system);
  font-size: 0.8rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.4rem;
}

.verse-selectors {
  display: flex;
  gap: 0.5rem;
}

.verse-selectors select {
  flex: 1;
  padding: 0.5rem;
  border: 1px solid rgba(200, 190, 170, 0.4);
  border-radius: 6px;
  background: var(--bg-warm);
  font-family: var(--font-system);
  font-size: 0.9rem;
  color: var(--text);
}

.prediction-text {
  font-family: var(--font-system);
  font-size: 0.9rem;
  color: var(--text-light);
  font-style: italic;
}

#report-notes {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid rgba(200, 190, 170, 0.4);
  border-radius: 6px;
  background: var(--bg-warm);
  font-family: var(--font-system);
  font-size: 0.9rem;
  color: var(--text);
  resize: vertical;
}

#report-audio {
  width: 100%;
  height: 40px;
}

.dialog-actions {
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
  margin-top: 1.25rem;
}

.report-status {
  font-family: var(--font-system);
  font-size: 0.85rem;
  color: var(--text-light);
  text-align: center;
  margin-top: 0.75rem;
}
```

**Step 5: Commit**

```bash
git add web/frontend/index.html web/frontend/src/report-dialog.ts web/frontend/src/main.ts web/frontend/src/style.css
git commit -m "feat: add error report dialog with audio playback and verse selector"
```

---

### Task 7: Add admin page

**Files:**
- Create: `web/frontend/server/admin.ts`
- Modify: `web/frontend/server/index.ts`

**Step 1: Create admin route module**

Create `web/frontend/server/admin.ts`:
```typescript
import { Hono } from "hono";
import { getCookie, setCookie } from "hono/cookie";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const STORAGE_DIR = process.env.STORAGE_DIR || "./storage/reports";
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || "tarteel-admin";

export const adminApp = new Hono();

// Auth check
function isAuthed(c: any): boolean {
  return getCookie(c, "admin_auth") === "1";
}

// Login page
adminApp.get("/login", (c) => {
  const error = c.req.query("error") ? "<p style='color:#c0564b'>Wrong password</p>" : "";
  return c.html(`<!DOCTYPE html>
<html><head><title>Admin Login</title>
<style>body{font-family:system-ui;background:#faf8f3;display:flex;justify-content:center;align-items:center;height:100vh}
form{background:#fff;padding:2rem;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);text-align:center}
input{display:block;margin:0.5rem auto;padding:0.5rem;border:1px solid #ddd;border-radius:6px;font-size:1rem}
button{margin-top:0.5rem;padding:0.5rem 1.5rem;background:#b8986a;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:1rem}</style>
</head><body>
<form method="POST" action="/admin/login">
  <h2>Admin</h2>${error}
  <input type="password" name="password" placeholder="Password" autofocus>
  <button type="submit">Login</button>
</form></body></html>`);
});

adminApp.post("/login", async (c) => {
  const form = await c.req.formData();
  const pw = form.get("password") as string;
  if (pw === ADMIN_PASSWORD) {
    setCookie(c, "admin_auth", "1", { path: "/admin", httpOnly: true, maxAge: 86400 });
    return c.redirect("/admin");
  }
  return c.redirect("/admin/login?error=1");
});

// Admin dashboard
adminApp.get("/", async (c) => {
  if (!isAuthed(c)) return c.redirect("/admin/login");

  let reports: any[] = [];
  try {
    const entries = await readdir(STORAGE_DIR);
    for (const entry of entries) {
      try {
        const raw = await readFile(join(STORAGE_DIR, entry, "meta.json"), "utf-8");
        reports.push(JSON.parse(raw));
      } catch { /* skip */ }
    }
    reports.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  } catch { /* empty */ }

  const rows = reports.map(r => `
    <tr>
      <td>${new Date(r.timestamp).toLocaleString()}</td>
      <td>Surah ${r.surah}, Ayah ${r.ayah}</td>
      <td>${r.modelPrediction || "—"}</td>
      <td>${r.notes ? r.notes.slice(0, 80) : "—"}</td>
      <td><audio controls src="/api/reports/${r.id}/audio" preload="none"></audio></td>
    </tr>`).join("");

  return c.html(`<!DOCTYPE html>
<html><head><title>Error Reports</title>
<style>
body{font-family:system-ui;background:#faf8f3;padding:2rem;max-width:1100px;margin:0 auto}
h1{color:#2c2416;margin-bottom:1rem;font-size:1.4rem}
.count{color:#8a7e6b;font-size:0.9rem;margin-bottom:1.5rem}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 6px rgba(0,0,0,0.06)}
th{background:#f5f0e8;color:#2c2416;padding:0.75rem;text-align:left;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.04em}
td{padding:0.75rem;border-top:1px solid #f0ebe3;font-size:0.9rem;color:#2c2416;vertical-align:middle}
audio{height:32px;width:200px}
tr:hover td{background:#faf8f3}
.empty{text-align:center;padding:3rem;color:#8a7e6b}
</style></head><body>
<h1>Error Reports</h1>
<p class="count">${reports.length} report${reports.length !== 1 ? "s" : ""}</p>
${reports.length ? `<table>
<thead><tr><th>Time</th><th>Expected Verse</th><th>Model Predicted</th><th>Notes</th><th>Audio</th></tr></thead>
<tbody>${rows}</tbody>
</table>` : "<p class='empty'>No reports yet.</p>"}
</body></html>`);
});
```

**Step 2: Wire admin into server**

Add to `web/frontend/server/index.ts`:
```typescript
import { adminApp } from "./admin";

app.route("/admin", adminApp);
```

**Step 3: Test admin page**

Visit `http://localhost:5000/admin` — should redirect to login.
Enter the password — should show the dashboard with any submitted reports.

**Step 4: Commit**

```bash
git add web/frontend/server/admin.ts web/frontend/server/index.ts
git commit -m "feat: add password-protected admin page for reviewing reports"
```

---

### Task 8: Update offline badge text

**Files:**
- Modify: `web/frontend/index.html`

**Step 1: Update badge**

Since reports are now sent to a server, the "100% offline" badge is no longer fully accurate. Update the `#offline-badge` text:

Change:
```html
100% offline — no data leaves your device
```
To:
```html
on-device recognition
```

The model still runs locally — recognition is offline. Only error reports are sent when the user explicitly submits.

**Step 2: Commit**

```bash
git add web/frontend/index.html
git commit -m "fix: update badge text to reflect report submission capability"
```

---

### Task 9: Add COOP/COEP headers to Hono server

**Files:**
- Modify: `web/frontend/server/index.ts`

**Step 1: Add required headers**

The Vite dev config sets `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` which are needed for `SharedArrayBuffer` (used by ONNX runtime WASM threads). Add these to the Hono server:

```typescript
import { cors } from "hono/cors";

// Add COOP/COEP headers for SharedArrayBuffer support
app.use("*", async (c, next) => {
  await next();
  c.header("Cross-Origin-Opener-Policy", "same-origin");
  c.header("Cross-Origin-Embedder-Policy", "require-corp");
});
```

**Step 2: Commit**

```bash
git add web/frontend/server/index.ts
git commit -m "fix: add COOP/COEP headers to Hono server for WASM threading"
```

---

### Task 10: Build, deploy, and test on Dokku

**Files:**
- Modify: `web/frontend/Dockerfile` (if needed)

**Step 1: Build and test Docker image locally**

```bash
cd /Users/rock/ai/projects/offline-tarteel/web/frontend
docker build -t tarteel-test .
docker run -p 5000:5000 -e ADMIN_PASSWORD=test123 -v /tmp/tarteel-reports:/app/storage/reports tarteel-test
```

Verify: `curl http://localhost:5000/api/health` returns `{"ok":true}`.

**Step 2: Set up Dokku persistent storage**

```bash
ssh dokku@dokku-server storage:ensure-directory offline-tarteel
ssh dokku@dokku-server storage:mount offline-tarteel /var/lib/dokku/data/storage/offline-tarteel:/app/storage
```

**Step 3: Set admin password env var**

```bash
ssh dokku@dokku-server config:set offline-tarteel ADMIN_PASSWORD="<choose-a-password>"
```

**Step 4: Deploy**

Push to Dokku (the exact git remote depends on current setup — check with `git remote -v`).

**Step 5: Browser test**

Use `/dev-browser` to verify:
1. App loads, Start button appears after model download
2. Click Start, speak, click Stop
3. Click Report Error, fill in surah/ayah, submit
4. Visit /admin, login, verify report appears with audio playback

**Step 6: Commit any fixes**

```bash
git add <files>
git commit -m "fix: deployment adjustments for Dokku"
```
