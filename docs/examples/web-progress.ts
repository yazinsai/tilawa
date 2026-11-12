// docs/quickstart.md: "Show download progress"
import { createTilawa } from "@tilawa/core/web";

const tilawa = await createTilawa({
  onProgress: (p) => {
    // p.stage is "model", "reference" or "warmup"; p.fraction is 0..1 within that stage.
    if (p.stage === "model") bar.value = p.fraction;
  },
});
console.log(`Loaded ${tilawa.model.id}@${tilawa.model.version} (${tilawa.model.tier}) on ${tilawa.model.backend}`);
