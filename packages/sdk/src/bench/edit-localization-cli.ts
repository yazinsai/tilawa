/** Diagnose a trained localization bundle on supplied engine evidence, without
 * choosing an operating point or reading planted labels. */
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "@tilawa/core";
import type { PhonemeEvidence, PhraseAsset, ReferenceAsset } from "@tilawa/core";
import { addEditLocalizationEvidence } from "../edit-localization.js";
import { loadNodeModel, readWav } from "../node.js";

const {values} = parseArgs({options:{model:{type:"string"},audio:{type:"string"},chunks:{type:"string"},out:{type:"string"}}});
if (!values.model || !values.audio || !values.chunks || !values.out) throw new Error("--model, --audio, --chunks and --out are required");
const model = await loadNodeModel(values.model);
try {
  if (!model.editLocalization) throw new Error("Model has no trained edit-localization declaration");
  const wav = await readWav(values.audio);
  if (wav.sampleRate !== model.manifest.sampleRate) throw new Error("Diagnostic audio must use the model sample rate");
  const asset = (file: string) => JSON.parse(readFileSync(new URL(`../../../core/assets/${file}`,import.meta.url),"utf8"));
  const index = loadReferenceIndex(asset("ref-v2.json") as ReferenceAsset), phrases = asset("phrases-v1.json") as PhraseAsset;
  const chunks = JSON.parse(readFileSync(values.chunks,"utf8")) as PhonemeEvidence[];
  if (!Array.isArray(chunks)) throw new Error("--chunks must contain the session's evidence array");
  const diagnostics: unknown[] = [];
  const enriched = await addEditLocalizationEvidence(model,wav.samples,chunks,index,phrases,(candidate,contextual,independent) => {
    diagnostics.push({...candidate,contextual,independent,
      contextualRejected:contextual < model.editLocalization!.preservation,
      independentRetained:independent >= model.editLocalization!.independentPreservation});
  });
  const primary = (es: PhonemeEvidence[]) => es.map(e => ({...e,phonemes:e.phonemes.map(({lexicalChanges:_,...p}) => p)}));
  if (JSON.stringify(primary(chunks)) !== JSON.stringify(primary(enriched))) throw new Error("Localization changed primary evidence");
  writeFileSync(values.out,JSON.stringify({primaryUnchanged:true,preservation:model.editLocalization.preservation,
    independentPreservation:model.editLocalization.independentPreservation,diagnostics},null,2));
} finally { await model.backend.release(); }
