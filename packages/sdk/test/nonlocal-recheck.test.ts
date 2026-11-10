import { describe, expect, it } from "vitest";
import type { LoadedModel } from "../src/model.js";
import { replayAcousticSlot, addNonlocalEvidence } from "../src/nonlocal-recheck.js";

describe("nonlocal acoustic word replay", () => {
  it("scores the complete written and rival readings on the same native frame interval", async () => {
    let frame = 0;
    const model = { manifest:{sampleRate:16000,frameMs:40,inventory:"test",tokens:"بتد",fbank:{mels:80,padValue:-16},
      window:11,shift:4,states:[]}, backend:{async run() {
        const winner = frame === 10 ? 1 : frame === 11 ? 2 : 0;
        frame++;
        return {log_probs:{type:"float32",data:Float32Array.from({length:4},(_,i) => Math.log(i === winner ? .97 : .01)),dims:[1,1,4]}};
      }}, supported:[] } as unknown as LoadedModel;
    const result = await replayAcousticSlot(model,new Float32Array(16000),{word:0,written:"بت",observed:"بد",fromFrame:10,toFrame:13},40);
    expect(result.phones).toBe("بت");
    expect(result.writtenNll).toBeLessThan(result.rivalNll);
    expect(Number.isFinite(result.rivalNll)).toBe(true);
  });
  it("does not enable an uncalibrated engine profile", async () => {
    const chunks = [{inventory:"test",frameMs:40,phonemes:[]}];
    expect(await addNonlocalEvidence({} as LoadedModel,new Float32Array(16000),chunks,{} as never)).toBe(chunks);
  });
});
