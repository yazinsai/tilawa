// docs/quickstart.md: "Errors"
import { MicrophoneError, ModelBudgetError, ModelIntegrityError, SessionEndedError, TilawaLoadError } from "@tilawa/core";
import { createTilawa } from "@tilawa/core/web";
import type { Tilawa } from "@tilawa/core/web";

async function load(): Promise<Tilawa | null> {
  try {
    return await createTilawa();
  } catch (e) {
    if (e instanceof Error && e.name === "AbortError") return null; // you cancelled it with `signal`
    if (e instanceof ModelIntegrityError) showMessage("The download was damaged. Please try again.");
    else if (e instanceof ModelBudgetError) showMessage("Not enough room to store the model on this device.");
    else if (e instanceof TilawaLoadError) showMessage(`Couldn't load the ${e.stage}. Check the connection and try again.`);
    else throw e;
    return null;
  }
}

const tilawa = await load();
if (tilawa) {
  const session = tilawa.startFreeRecitation();
  try {
    await tilawa.listen(session);
  } catch (e) {
    session.dispose();
    if (e instanceof MicrophoneError) {
      if (e.reason === "denied") showMessage("Allow the microphone in your browser settings, then try again.");
      else if (e.reason === "insecure-context") showMessage("The microphone needs https (or localhost).");
      else showMessage("No microphone was found.");
    } else if (!(e instanceof SessionEndedError)) throw e;
  }
}
