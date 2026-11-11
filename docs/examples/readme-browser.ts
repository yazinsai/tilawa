// The browser snippet in the root README.md, kept here so it is typechecked too.
import { createTilawa } from "@tilawa/core/web";

// Downloads the model and the Quran reference once, then serves them from the Cache API.
const tilawa = await createTilawa({ onProgress: (p) => (bar.value = p.fraction) });

const session = tilawa.startFreeRecitation();
session.subscribe((e) => {
  if (e.type === "position" && e.observed[0]) highlight(e.observed[0].word); // { surah, ayah, word }
  if (e.type === "checker" && e.state === "confirmed") mark(e.location, e.errorClass);
});

const mic = await tilawa.listen(session); // microphone → session
// …
await mic.stop();                         // stops the tracks, then ends the session
console.log(session.mistakes());
tilawa.dispose();                         // releases the ONNX session
