# Listener selected-checkpoint export

Status: killed on the source-level absence of the proposed stale export
snapshot. Deployed checkpoint identity remains unverified. The experiment card
is unchanged.

## Prediction and mechanism

Predict one recovered addition, listener −1 and overall −1, with zero counted
losses. The proposed defect is export retaining separately initialized or
previously copied parameters instead of the selected trained checkpoint's
named state, including checkpointed buffers. A repair would reconstruct the
inference model from that snapshot and bind the generated artifact's metadata
to it, preserving quantization, streaming geometry, vocabulary and warning
rules.

The discriminator is a parameter-lineage discrepancy, not ordinary fp32 versus
int8 recognition differences. The repaired artifact must be compared with the
original export of the same checkpoint, without retraining or selecting a
checkpoint on practice results. The task also requires stopping if a built
change recovers fewer than three targets. No such repair or acoustic arm was
produced because the checked-in path already performs the proposed state
restoration.

## Source audit

The audit uses this worktree at `a40e29a4` and the pinned training environment
(PyTorch 2.8.0; icefall `3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`).

- `modal/student_train.py:442–446` saves `model.state_dict()` together with
  optimization state and configuration to `ckpt.pt.tmp`, atomically replaces
  `ckpt.pt`, and commits the volume. It saves the current live model rather than
  a separately retained inference model. The procedure has one run checkpoint;
  there is no best-versus-last selector in this export path.
- `load_student` at lines 497–503 reads that checkpoint once, constructs a fresh
  model using its saved size, and restores the entire `ck["model"]` mapping.
  `load_state_dict` uses its default strict behavior. It does not filter names,
  restrict restoration to the encoder or head, ignore missing state, or fall
  back to initialized weights. The mapping includes all parameters and
  persistent registered buffers saved by `state_dict`. Nonpersistent buffers
  are not entries in the selected checkpoint snapshot.
- `modal/student_export.py:143–147` passes this loaded model straight to
  `export_onnx`. `wrap` at lines 44–71 registers the same `encoder`,
  `encoder_embed` and `ctc_output` module objects. Those are all three child
  modules constructed by `build_model`; the wrapper creates no replacement
  encoder or head and copies no cached parameter snapshot. Evaluation mode and
  streaming chunk/context settings are applied to these loaded modules.
- `export_onnx` exports this wrapper. Its temporary encoder forward-method
  substitution changes the callable used for tracing, not the modules supplying
  parameters. A second `load_student` at line 148 supplies the later PyTorch
  parity reference, after the ONNX graph has already been written. A concurrent
  checkpoint replacement between those reads could invalidate that comparison,
  but cannot make the graph retain a separate stale model. No such concurrent
  replacement was demonstrated here.
- The fp32 manifest receives the first load's checkpoint and records its
  training step plus the exported graph's SHA-256. `tools/quantize_onnx.py`
  reads that graph, dynamically quantizes MatMul weights, and records both the
  output hash and `quantizedFrom` source-graph hash. It does not read or select
  another training checkpoint. The SDK consumes the supplied ONNX artifact;
  it does not reconstruct student parameters from a checkpoint.
- The existing auxiliary teacher exporter in
  `modal/student_teacher_export.py` wraps `teacher.model` itself after loading
  and setting evaluation mode. It has no separately initialized export copy.
  This path is distinct from the student checkpoint under investigation.

The manifests do not contain a hash of the named checkpoint snapshot. Training
step and graph hashes therefore do not independently certify checkpoint identity.
That provenance limitation does not establish a parameter mismatch or justify
changing weights or manifest schema under this card.

## Validation and limits

Local AST checks passed for the fresh-model construction before strict
full-state restoration, direct module registration in the streaming wrapper,
and export occurring between the two checkpoint reads. These are checks of the
checked-in control flow, not numerical export-parity or acoustic assertions.

Supervisor run 1 requested inspection of the supplied model's files and
provenance fields, runtime dependency availability, and checkpoint availability
under the authorized `LAB_DATA` directory. Run 2 requested the supplied
six-shard core/SDK build/test harness. Both returned
`NotFoundError: No such file or directory`, with no command logs. No artifact
inspection, numerical parity result or test pass was obtained. Infrastructure
failures are not passed assertions or test failures in the implementation.

No actual selected checkpoint was compared with the deployed graph, and none
of the five target audio spans was replayed. This audit cannot certify unseen
historical exporters, external checkpoint rewrites, dependency internals at
runtime, or recognition quality. It does not establish whether quantization
preserves a small trained correction; that is a different mechanism.

## Outcome

Reconstructing the inference model from the selected checkpoint is already the
current loader's behavior. No demonstrated stale parameter or omitted
checkpoint buffer warrants a repair. Stop on that source-level premise instead
of adding speculative export hardening or training another model.

No engine behavior changed, so no behavior-specific unit test is retained and
no engine rollback is needed. No training, replacement bundle, model declaration,
target gain measurement or practice score was performed (0 of 3 scores used).
`targets won: 0` in `.autolab/outcome.json` records that this branch delivers no
changed listener, not a measured zero-gain acoustic comparison. No overall or
listener score delta is claimed.

Only this report is committed. The ignored outcome file records the killed
result. No scorer, lab tooling, supplied manifest, test data, evidence,
experiment-card, tracker, checker or SDK changes were made. No fold occurred.
