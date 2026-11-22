# Live microphone review

Open the lab's **Live mic** tab (`/live/`). Start the microphone, recite, and watch the listener's latest phonemes, the tracker's ayah and word, and the checker's warning states. The model downloads once and is cached in the browser. Use HTTPS and allow microphone access.

This page imports the existing core and SDK without changing their policies. It uses the registry's full INT8 listener (`spliced-clean-int8`, the lab default), with free recitation and no VAD so audio and diagnostic timestamps agree. Inference runs on the reviewing device. The header identifies the exact source revision bundled at deployment; it is not a continuously updating view of Git.

Click **Flag issue** on any component. A flag captures that moment; keep reciting while writing the note. Stop to save flagged sessions automatically. After stopping, seek the playback to a moment and add another flag, then click **Save flagged session**. Each take is limited to three minutes. A failed upload leaves the recording available for retry; do not close the tab or start a new take until it saves.

Audio is uploaded only for flagged takes. On the `tilawa-lab` Modal volume, `/reviews/live/<session UUID>/` contains:

- `audio.wav`: mono 16 kHz audio actually submitted to the engine.
- `report.json`: build and model identity, browser details, timestamped flags, events, and listener evidence.
- `issues/NN-<component>.md`: an issue per flag, initially `Status: needs-triage`.

Download the folder with `modal volume get tilawa-lab reviews/live/<session UUID> .scratch/live-review/<session UUID>`. The UI also offers diagnostic JSON export, and saved reports/audio can be retrieved at `/api/live/sessions/<session UUID>/report.json` and `/api/live/sessions/<session UUID>/audio.wav`.

## Deploying the current main

From the main checkout:

```sh
python3 tools/build_live_review.py
modal deploy modal/lab_review.py
```

The build refuses tracked engine modifications or a branch other than main, verifies the registered model checksum, and packages the browser runtime, model, and UI. It expects the full model export under `.scratch/models/` and installed workspace dependencies. Rebuild before each deploy to update the bundled engine. No benchmark experiment is required for this UI-only change: listener, tracker, and checker implementation and policies are unchanged.

## Checks

```sh
pnpm --filter @tilawa/sample exec tsc --noEmit
python3 -m unittest discover -s tools -p test_live_review.py
python3 tools/build_live_review.py
```
