# Word-Level Highlighting Design

## Problem
Once a verse is identified, the UI highlights the entire verse. The user wants to see individual words light up as they're spoken, with minimal lag.

## Approach
After a verse match, switch to a fast "tracking mode" that transcribes every 0.5s and aligns the transcription against the known verse text using LCS. Send word-position updates to the frontend, which renders each word as its own `<span>`.

## Server Changes (web/server.py)

Two modes in the WebSocket loop:

1. **Discovery mode** (existing) — trigger every 2s, search all verses
2. **Tracking mode** (new) — trigger every 0.5s (8,000 samples), run LCS alignment against known verse, send `word_progress` messages

Tracking mode exits when:
- Word position reaches end of verse → auto-advance to next verse
- Alignment diverges → fall back to discovery
- Extended silence → reset

New WebSocket message:
```json
{
  "type": "word_progress",
  "surah": 2, "ayah": 255,
  "word_index": 5, "total_words": 50,
  "matched_indices": [0, 1, 2, 3, 4, 5]
}
```

Alignment logic: LCS-based DP from verse_position_tracker.py's `_align_position`.

## Frontend Changes (web/frontend/src/main.ts)

- Split each verse's text_uthmani into per-word `<span class="word">` elements
- On `word_progress`: mark matched word spans with `word--spoken` class
- On `verse_match`: same as today (new surah render) plus per-word setup

## CSS Changes (web/frontend/src/style.css)

Active verse words start muted, light up on match:
```css
.word { transition: color 0.15s ease; }
.verse--active .word--spoken { color: var(--text); }
.verse--active .word:not(.word--spoken) { color: var(--text-muted); }
```

## Performance
- Tracking trigger: 0.5s of new audio
- FastConformer inference: ~0.33s
- Update rate: ~1 per 0.8-1s
- LCS alignment: negligible
