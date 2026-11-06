# Word-ear candidates judged by teacher insertion arcs

Status: implemented; three of six registered targets pass the teacher gate in
audio-backed replay. Practice replay is unchanged; the supervisor's fresh
audio-backed decode and fold gate remain pending. The experiment card is unchanged.

## Prediction and mechanism

Predict listener −2 and overall −1. Stop on the premise if fewer than three
registered targets have a word-ear extra word confirmed by a teacher arc
marginal strictly above 0.5. Aggregate losses belong to the fold gate.

Reuse the portable int8 Whisper-small word ear from
`listener-short-crop-word-ear-seams`, without retraining or confidence gating.
It reads detector-pause to tracker-word-two seam crops. The read-only clock
recognizes ayah and mushaf-marked inner boundaries and finds the eventual
word-two step through intervening tracker events. No labels choose engine
spans or candidate identities.

A sequence diff preserves written words matched in order and proposes unmatched
heard words. A proposal must map to a whole mushaf word pronunciation and pass
the existing lexical neighbour, prefix, back-run and local-fragment exclusions.
Formula words are excluded. Autoregressive log-probability is ignored.

Each remaining word is judged by the unchanged folded `seamInsertionMarginal`:
previous two words, an optional exact word arc entered at `PHRASE_SWITCH`, next
two words, free CTC boundaries and contiguous repeats. The existing student
exact-spelling path remains available. Teacher-certified word-ear proposals are
transported separately on the existing first-token evidence certificate route,
so they need no student spelling, ownership, support margin or confidence.
Primary phonemes, posteriors and tracker input are preserved. Audio-backed
sessions publish the additional confirmed warnings at completion.

The normal Node student loader resolves the declared encoder, decoder and byte
tokenizer adjacent to the teacher. Both word backends load lazily and are
released with the primary model. Missing ear assets keep the existing teacher
path. A partially installed ear reports an error.

## Reused assets and prerequisite correction

The precursor's retained write-up records **six raw whole-word recoveries across
ten targets**, rather than six recoveries on this card's six registered targets.
Its recovered identities were p48, p50, p55, both p57 sites, and the older p19
site. It did not recover p07 or p56 and found no eligible p33 pause span. This
experiment therefore measures the current engine spans and teacher decisions;
the six raw precursor recoveries are not six confirmed additions.

Modal run 2 copied the original assets into this experiment's scratch bundle,
along with unmodified primary model and teacher files. No supplied manifest,
audio, evidence, label, scorer or lab-tool source was edited.

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| word-encoder.onnx | 99,236,529 | ad4ab2311b9a43c74cc322ed3956001630a2a11a22d3de747e7a29e0741c6319 |
| word-decoder.onnx | 315,826,781 | d03c3d61272ffc7b009bcf8b9ddd85ee357d14c9eaacb8d096aa17330dc7bbd5 |
| word-tokenizer.json | 627,413 | 28887b7ba1457da64400f0479ad28d2b4c6cfa56e7fe5355349479296d60774a |

The reused ear adds approximately 416 MB to the folded fp32 teacher's 2.43 GB.
Buffering session audio and deferring judgments until completion are additional
deployment limitations. This experiment evaluates the acoustic mechanism, not
device-budget or streaming-latency readiness.

## Validation and results

Unit coverage includes split UTF-8 and BPE reconstruction, autoregressive EOS
and truncation, pause/step seam discovery, whole-word diff and lexical guards,
lazy asset loading and release, primary-value preservation, and a candidate-free
student whose arbitrarily low-confidence word-ear proposal is accepted only on
teacher extra-speech rows. Neighbour-only teacher rows reject the same proposal.
Core replay checks the strict 0.5 cut, certificate provenance, deduplication and
neighbour rejection independently of student confidence.

Initial Modal runs 3 and 4 stopped on two checker helper scope errors. Those
were fixed without changing acoustic rules. Run 5 built successfully; 35 SDK
tests passed, one optional test was skipped, and one release-spy assertion
failed because the loader wrapped the backend method. The assertion now retains
the original spy. No existing enforcement or timeout was relaxed.

Run 7 passed all 17 focused SDK/core tests. Runs 7–9 did not execute acoustic
diagnostics because the supervised snapshot omitted the temporary probe, even
after it was moved and staged. Run 11 embedded the probe in the submitted
command but stopped before inference because the snapshot also omitted the
experiment card. Run 13 supplies the card's listener identifier through an
environment variable instead. These diagnostic fixes change no engine behavior.
Run 6 passed the supplied six-shard harness: **758 core
tests and 36 SDK tests**, with one optional SDK test skipped. Every shard exited
zero. The model loader, teacher marginal, primary listener/session and existing
tracker/checker tests all passed without relaxing enforcement.

Run 13 completed all five diagnostic shards successfully. It loaded the actual
portable word ear and fp32 teacher through the normal Node loader. Engine span
and candidate discovery read no labels; labels selected/report the diagnostic
sites, including other added-word sites in those five recordings. Unrelated
word-ear calls returned no proposals to reduce diagnostic compute. The existing
student exact-candidate route still ran. Every recording preserved primary
phonemes and posterior objects, and tracker events were byte-identical after
JSON serialization and replay.

| Registered target | Representative ear transcript | Best eligible teacher marginal | Confirmed addition |
| --- | --- | ---: | --- |
| p07, 37:36:1 | سليم ويقولون أن | 1.0 | yes, سَلِۦم |
| p33, 26:208:1 | no eligible pause/step span | — | no |
| p48, 12:76:1 | كذب فبأى | 0.00000822352 | no |
| p56, 32:21:1 | الطين ولنذيق عنهم | 0.999999999826 | yes, ططِۦنِ |
| p57, 51:18:1 | قبل وبالأسحار هم | 0.999996638200 | yes, قَبلُ |
| p57, 51:20:1 | قبل | 0.00000000000522964 | no |

Some sites have two detector-pause starts. At p07 the earlier start writes سليم
and the later start writes سبيل; at p56 only the later start writes an eligible
word. Neither alternative was chosen by a label or a confidence cut. Both are
ordinary engine-discovered spans. The teacher rejects the p48 and second p57
proposals despite their whole-word identity recovery. Thus naming a word does
not imply that the unchanged closed teacher lattice will certify it.

Three registered targets receive teacher-certified confirmed warnings, meeting
the prerequisite to continue to practice scoring. These are diagnostic target
results, not a measured overall or component score. The probe is retained only
in ignored local `.autolab` files; it is not part of the engine or scoring path.
`.autolab/model.json` declares the scratch `bundle` so the supervisor can decode
with all reused assets. No supplied manifest is modified.

Read-only route inspection confirms that `serve_practice` invokes `lab run`,
which replays the ledger's recorded listener evidence. It does not copy or
decode the new bundle. The supervisor's later listener validation copies every
bundle file and forces phone/studio decodes through the SDK audio session with
`recordEvidence: true`; its output retains the auxiliary certificates. Therefore
the requested practice replay and the audio-backed diagnostic measure different
available evidence until that fresh decode occurs. No scorer or decode-tool
change is needed or made to generate these certificates in the normal SDK path.

Practice score 1 completed at implementation commit `c577a5b8`:

| Charged component | Main | Branch | Change |
| --- | ---: | ---: | ---: |
| Overall | 24 (13 missed, 11 false) | 24 (13 missed, 11 false) | 0 |
| Listener | 23 (13 missed, 10 false) | 23 (13 missed, 10 false) | 0 |
| Tracker | 1 (0 missed, 1 false) | 1 (0 missed, 1 false) | 0 |
| Checker | 0 | 0 | 0 |

The verdict is noise, with no practice gain claimed. This requested replay has
no newly generated word-ear certificates, so it does not reproduce the three
audio-backed target confirmations. The aggregate result does not satisfy the
fold gain requirement. The teacher-marginal prerequisite nevertheless passes
on three registered targets, and no loss-based narrowing is justified by this
unchanged replay. The implementation and declared bundle remain in place for
the supervisor's forced fresh listener decode and gate; no killed outcome is
written and no fold is performed here. One of the three permitted practice
scores was used. No further engine changes followed the passing tests.
