// docs/quickstart.md: "Web in ten lines"
import { createTilawa } from "@tilawa/core/web";

const tilawa = await createTilawa(); // downloads the model and the Quran reference once, then caches them
const session = tilawa.startFreeRecitation();
session.subscribe((e) => {
  if (e.type === "position" && e.state === "locked" && e.observed[0]) highlight(e.observed[0].word);
  if (e.type === "checker" && e.state === "confirmed") mark(e.location, e.errorClass);
});
const mic = await tilawa.listen(session); // asks for the microphone, then streams it into the session

stopButton.onclick = async () => {
  await mic.stop(); // stops the microphone and ends the session
  await tilawa.dispose();
};
