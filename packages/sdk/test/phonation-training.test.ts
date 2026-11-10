import { describe, expect, it } from "vitest";
import { selectPhonationTrainingLabels } from "../../../modal/student_phonation_prepare.js";

const clean={speaker:"training",kind:"none",split:"train",scored:true,events:[],non_errors:[],unsupported:[]};

describe("phonation training label eligibility",()=>{
  it("accepts importer rows with an absent exclusion marker and explicit false markers",()=>{
    const labels=[clean,{...clean,excluded:false}];
    expect(selectPhonationTrainingLabels(labels).rows).toEqual(labels);
  });
  it("rejects excluded, unscored, scripted, repeated and non-training recordings and speakers",()=>{
    const labels=[clean,{...clean,excluded:"audit"},{...clean,excluded:true},{...clean,scored:false},
      {...clean,kind:"wrong_word"},{...clean,events:[{}]},{...clean,non_errors:[{}]},
      {...clean,unsupported:[{}]},{...clean,split:"eval"}, {...clean,speaker:"eligible"}];
    const result=selectPhonationTrainingLabels(labels);
    expect(result.excluded.has("training")).toBe(true);
    expect(result.rows).toEqual([{...clean,speaker:"eligible"}]);
  });
});
