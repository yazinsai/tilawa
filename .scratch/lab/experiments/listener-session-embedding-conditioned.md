# Causal session embedding conditioned student

Status: implemented and validated, left selected for the supervisor's custom-model fold gate. Practice request 1 was a +0 replay of the card's stored listener, not a score of this model. Four reproduced target warnings disappear. Three candidate slots spell the written word, but two already do so in the fresh control. The experiment card is unchanged.

## Prediction and continuation check

Predict three of seven target slots recovered, about one caught swap lost:
listener −2, overall −1. Continue to practice scoring only if the built int8
student, decoded as a full session, spells at least three target slots as the
written word with zero hard edits under the existing soft costs. Warning
removal and written-word recovery are separate measurements.

## Mechanism

A frozen copy of the shipped spliced-clean student supplies stack 3 features,
after upsampling and residual combination, on the 20 ms grid. A small
LayerNorm/MLP maps each frame to 64 dimensions. The running mean includes every
completed chunk, including silence and opening recitation. The next chunk
receives that mean once at least one second has accumulated. There is no
reference text, tracker output, gradient update or fitted session parameter in
this input. The chunk scheduling makes the condition independent of the
caller's audio push size.

Each BiasNorm in the acoustic student has a bias-free linear projection of the
session vector to scale and offset. Its weights start at zero, so initialization
is identity; a zero vector also makes every FiLM operation identity after
training. The acoustic weights may still change in training.

Training uses clean-match/train clips, at most 30 seconds long, sampled to
20 hours with seed zero. Every voice cluster containing a gold-excluded clip
is excluded entirely. A target's context always comes from a different clip
of the same remaining cluster, with a prefix duration sampled between one
second and the clip's full duration. Both clips use the same seed through the
shipped noise/RIR/gain/SpecAugment chain and the parent's saved probabilities.
The supervision is canonical collapsed-phone CTC and the original teacher's
KD, each at weight 1. Every fourth item zeros the condition and adds KL-to-shipped at weight
10. AdamW uses 0.00003 for acoustic weights and 0.0003 for FiLM and the frame
encoder, for a fixed 3,000 steps; checkpoint resumes preserve optimizer and
random states. No training setting is selected using practice.

The graph exports the frozen and conditioned streaming branches, their
separate recurrent states and the learned frame features. Export uses the
existing streaming window (45 frames), shift (32), context (128), and dynamic
MatMul int8 quantization. The SDK maintains the running sum and count per
listener, excluding EOS padding and overlapping lookahead. The checker,
tracker and their constants are unchanged.

## Validation and limits

Supervisor run 1 verified availability of the shipped parent and the s1
schema, including `cluster`, canonical reference, teacher posteriors and
16 kHz FLAC audio. tlog has no verified speaker identities: its clusters are
approximate voice groups, so same-cluster pairing is not proof of same-person
pairing. This is a limitation of this training test. The synthetic augmentation
chain is unchanged; this experiment adds no new phone transform or corpus.

The frozen branch increases inference compute and recurrent-state memory.
Mobile latency and footprint need measurement before a release claim. A KL
anchor discourages output drift but does not guarantee unchanged strings.
No independent listening review or label correction is part of this experiment.

Runs 2 and 3 failed in Modal's volume infrastructure (`too many layers in
volume`); neither returned a usable training/test result. Run 4 reached the
six-shard build but found a pre-existing TypeScript error in `reference.ts`:
`filter(Boolean)` retained `undefined` in the inferred type of optional plain
onset forms. The explicit type predicate preserves that filter's runtime
behavior and changes no pronunciation form.

Run 6 passed core and SDK builds and all six validation shards: 737 core tests
and 17 SDK tests passed, with one existing optional SDK test skipped. The
unchanged test wrapper accepted the known Vitest RPC timeout only after all
core tests in that shard passed. The new SDK case checks the one-second gate,
completed-chunk running mean, silence inclusion, push-size independence,
per-listener reset and rejection of missing feature output.

Run 6 also read the persistent pool audit: 8,445 clips from 90 training voice
clusters, sampled to 20.0002 hours, after excluding all 48 clusters linked to
the gold clip exclusions. At that snapshot only data preparation was visible;
no completed optimization checkpoint was yet reported.

Run 7 passed all three Python tests, including a small actual Zipformer stream:
64-dimensional frozen features on the 20 ms grid, unchanged initial logits,
separate recurrent states, frozen parameters, zero-vector identity after
nonzero FiLM weights, and both BiasNorm batch layouts. Run 8 read a valid
2,000-step checkpoint from the persistent scratch area. Open log files were
not visible in that concurrent container's snapshot.

Run 5 completed the fixed 3,000-step training run with finite objectives and
gradients. Optimization took 1,378.2 seconds; the supervisor command took
1,855.2 seconds including preparation. GPU cost was $2.04 (plus $0.50 spent
on the earlier infrastructure failure), leaving $117.46. The final minibatch
logged CTC 0.09661, KD 0.24614, embedding norm 0.54325 and FiLM weight norm
12.1902. Logs are sampled every 50 steps and therefore miss the every-fourth
zero-conditioned items; their printed KL values are zero and must not be read
as an aggregate KL measurement. The zero-conditioned term is computed on the
intervening items by the fixed training loop.

Parent checkpoint SHA-256:
`bcd41fcffe1b298cb4da36ea0e3df87d1033e209b8bba1cbff7d0a654b749b9f`.
No intermediate checkpoint was selected using target or practice results.

Run 9's export preflight read the final 3,000-step checkpoint. It produced an
int8 graph but failed its Torch/float-ONNX parity assertion (maximum log-prob
error 4.1–5.9 and feature error 0.025–0.040). Inspection identified a validation
mode bug: the outer `Streaming` wrapper defaulted to training mode, so ONNX's
mode restoration re-enabled dropout on the reused Torch oracle after export.
The inference wrapper now starts in evaluation mode; the small-model test
asserts that every submodule stays in evaluation mode. Parity state arrays
also receive independent copies rather than sharing Torch memory. This is an
export/validation fix, not a new training arm or a target-driven adjustment.

Run 10 passed all three Python tests with the mode regression assertion and
exported the final checkpoint with `--require-complete`. It reproduced run 9's
int8 graph hash, confirming that the failed parity check concerned the Torch
oracle rather than a different exported graph. On three training-pool clips,
float streaming log-probability errors were at most 0.0000191 and feature
errors at most 0.000000716. Int8 top-token agreement against the float graph
was 99.39%, 100% and 99.32% (165, 87 and 441 frames). The int8 feature feedback
uses its own running mean, as the SDK does.

Final persistent export: `model-int8/` under this experiment's scratch area;
136,252,537 bytes. SHA-256:
`71f8fc19ab1bc01300eb4b64e6975ede569cf672c25329eab0e8f73ca74f74d3`.

Runs 11 and 12 freshly decoded the seven complete phone recordings with the
shipped and final conditioned int8 graphs, using the SDK's final gap-spliced
evidence. Run 13 replayed the controls with a passive slot inspector. On the
current checker, only four historical target warnings reproduce even from
the supplied stored baseline evidence: p11, p36, p55 and sob-049. The other
three are absent and cannot be counted as warnings removed by this model.

The first inspector required a confirmed baseline accusation and therefore
aborted on those three absent targets. The completed four candidate checks
all removed their original warning, but only p55 spelled the written word
with zero hard edits (p11: two, p36: one, sob-049: one). This is a preliminary
four-slot result, not the seven-slot continuation verdict. The missing cases
are checked by choosing the baseline's counting visit nearest the target time
plus the existing 2.5-second confirmation delay, before reading edit counts.
No phonetic score is used to choose a span. Baseline absence is retained in the
output, and these cases cannot inflate the warning-removal count.


## Complete target check

Run 14 checked all seven completed full-session streams and both fresh
controls, without further decoding, weight changes or threshold selection.
Passive tracing reproduced every original checker event in every replay.
For the three absent baseline accusations it chose the counting visit nearest
the historical target time plus the unchanged 2.5-second delay. The choice is
made by time, not phonetic score, and identical baseline/candidate spans are
used where the word alignment does not change.

| Target | Fresh control hard edits | Candidate hard edits | Original warning reproduced | Warning removed |
| --- | ---: | ---: | --- | --- |
| p11 / 26:106:1 | 2 | 2 | yes | yes |
| p36 / 55:48:1 | 2 | 1 | yes | yes |
| p55 / 26:174:8 | 5 | 0 | yes | yes |
| sob-027 / 84:25:1 | 2 | 2 | no | not a gain |
| sob-049 / 92:13:1 | 2 | 1 | yes | yes |
| sob-053 / 88:2:2 | 0 | 0 | no | not a gain |
| tlo-127 / 2:29:16 | 0 | 0 | no | not a gain |

The literal card criterion is met: three candidate slots have zero hard edits.
Only **one is newly restored**; the other two were already spelled correctly
and quiet in the fresh control. This does not validate the prediction of three
new phonetic recoveries. Four currently reproduced warnings disappear, so the
user's continuation instruction requires practice scoring before a decision.
The fixed trained graph is retained for that evaluation and the supervisor's
fold gate; the control result is not used to invent a stricter stopping rule
or to count already-correct words as gains.

The only restored word, p55's مؤمنين, has a complete candidate slot of frames
2491–2549 rather than the baseline's 2491–2656 glued surplus. p36 recovers its
initial ذ but retains ف in place of ت. sob-049 recovers ن but retains surplus
ي. p11 still has the wrong glide. Warning disappearance therefore mostly
reflects changed confidence, margins or alignment, rather than all the missing
written consonants being recovered.

The temporary diagnostic source and target-input files have been removed.
Practice score request 1 uses committed model selector `model-int8` and commit
`3fd6f290`; the complete validated implementation is preserved on this branch.

No zero-embedding practice ablation was selected or scored. Because acoustic
weights also fine-tuned, these observations cannot isolate the benefit of the
session vector from changes shared by all sessions. The target checks do not
measure caught-swap retention, fully clean sessions, studio behavior or
reader-excluded gains. Those remain the fold gate's responsibility.

## Practice scoring path limitation

Read-only inspection of the current supervisor confirms the previously
reported custom-model limitation: `serve_practice` calls `lab run` directly;
`run_experiment` chooses `card["listener"]`, which remains the original
`exp-listener-stack-be988c-6a17f391` as instructed. That path does not read the
new model selector. Only the supervisor's later `after_tests` path reads
`.autolab/model.json`, copies the persistent model and forces new phone/studio
decodes before scoring with the new listener. No scorer, tooling or card was
changed to bypass this. The requested score can validate the unchanged
engine on stored baseline evidence, but cannot establish this model's overall,
clean-session, studio or incident trade. Repeating the same request would
not resolve that limitation.


Practice request 1 completed with **30 wrong calls** (20 missed, 10 false),
matching its baseline: overall +0, listener +0 (29 wrong calls), tracker +0
(one) and checker +0 (zero), verdict noise. It scored commit `3fd6f290` through
the stored-listener path described above. This is not a measured overall
change for the conditioned graph. One of three practice requests was used;
no repeat was requested because the same path would use the same listener.

The literal three-slot spelling check passes and four reproduced warnings
are removed. The implementation and final model selector remain in place for
the supervisor's actual custom-model decode and fold gate. No killed outcome
is written, no fold is requested, and no overall gain, clean-session safety,
studio safety or planted-swap retention is claimed. The target gain is four
removed warnings and one newly restored spelling, not three new recoveries.
All engine code, tests, model selection and this write-up are committed. The
card, scorer, lab tooling, test data, existing manifests and evidence are
unchanged. Generated weights and export files remain in the experiment's
persistent scratch area and are selected by `.autolab/model.json`.
