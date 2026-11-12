// docs/quickstart.md: "iOS Safari: open the microphone inside the tap"
import { createTilawa, prepareMicrophone } from "@tilawa/core/web";

startButton.onclick = async () => {
  // Synchronously, before any await: iOS only starts audio created during the tap.
  const prepared = prepareMicrophone();
  try {
    const tilawa = await createTilawa(); // may take a while on the first visit
    const session = tilawa.startFreeRecitation();
    session.subscribe((e) => {
      if (e.type === "position" && e.state === "locked" && e.observed[0]) highlight(e.observed[0].word);
    });
    const mic = await tilawa.listen(session, { prepared }); // listen() takes ownership of `prepared`
    stopButton.onclick = () => void mic.stop();
  } catch (e) {
    prepared.release(); // safe to call twice
    throw e;
  }
};
