import { describe, expect, it } from "vitest";
import { recurrenceFeatures, recurrenceVeto, scoreRecurrence, validateRecurrenceModel, type RecurrenceModel } from "../src/acoustic-recurrence.js";

const model = (): RecurrenceModel => ({ schema:1,inventory:"test",tokens:"بت",frameMs:40,dimensions:1,
  center:[0],scale:[1],means:[[0],[0],[1]],variances:[[1],[1],[1]],
  pairMeans:[[0,0,0],[0,1,-1],[0,-1,1]],pairVariances:[[1,1,1],[1,.1,.1],[1,.1,.1]],
  nullMean:0,nullVariance:1,weight:1,beam:64,maxFrames:256 });

describe("within-occurrence recurrence scoring",()=> {
  it("marginalizes every bounded phone placement, retaining complete repeated phones",()=> {
    const m=model();m.weight=.5;
    const result=scoreRecurrence(m,[[1],[1],[1],[1]],"ببت")!;
    expect(result.paths).toBe(3); // all compositions of four frames into three nonempty states
    expect(scoreRecurrence(m,[[1]],"بب")).toBeNull();
    expect(scoreRecurrence(m,[[1]],"x")).toBeNull();
  });
  it("uses no relationships on adjacent frames",()=> {
    const score=scoreRecurrence(model(),[[1],[1]],"بت")!;
    expect(score.relational).toBe(score.local);
    expect(score.positive).toBe(0);
    const restricted=scoreRecurrence(model(),[[1],[1],[1],[1]],"بت",[1,3])!;
    expect(restricted.relational).toBe(restricted.local);
    expect(scoreRecurrence(model(),[[1],[1]],"بت",[-1,2])).toBeNull();
  });
  it("distinguishes nonadjacent repetition with a matched local emission control",()=> {
    const m=model();m.means[1]=[0];m.means[2]=[0];
    const repeated=scoreRecurrence(m,[[1],[-1],[1]],"بتب")!;
    const different=scoreRecurrence(m,[[1],[-1],[1]],"بتت")!;
    expect(repeated.local).toBe(different.local);
    expect(repeated.relational).toBeGreaterThan(different.relational);
    expect(repeated.positive).toBeGreaterThan(0);
  });
  it("requires reversal and affirmative written evidence, abstaining on ties or unavailable scores",()=> {
    const written={local:-2,relational:1,positive:3,paths:1};
    const accused={local:-1,relational:0,positive:1,paths:1};
    expect(recurrenceVeto(written,accused)).toBe(true);
    expect(recurrenceVeto({...written,positive:0},accused)).toBe(false);
    expect(recurrenceVeto({...written,local:-1},accused)).toBe(false);
    expect(recurrenceVeto({...written,relational:0},accused)).toBe(false);
    expect(recurrenceVeto(null,accused)).toBe(false);
  });
  it("extracts on the primary clock and rejects invalid assets",()=> {
    expect(recurrenceFeatures(new Float32Array(16000))).toHaveLength(24);
    expect(validateRecurrenceModel(model(),"test","بت")).toEqual(model());
    expect(()=>validateRecurrenceModel({...model(),scale:[0]},"test","بت")).toThrow();
    expect(()=>validateRecurrenceModel(model(),"other","بت")).toThrow();
  });
});
