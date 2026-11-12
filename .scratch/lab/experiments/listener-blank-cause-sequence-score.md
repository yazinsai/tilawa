# Listener blank-cause sequence scoring

Status: implementation prepared; remote training and validation blocked. No measured target result or practice score. The experiment card is unchanged.

## Prediction

The registered prediction is listener −1 and overall −1, removing one false wrong-word warning with no predicted losses. The card requires at least two targets whose rival-favouring complete pronunciation paths reverse under the training-frozen two-cause blank model. The task's stricter stopping rule requires at least three targets actually removed before a full practice score. Neither condition has been measured.

## Mechanism

The SDK extracts log energy, normalized spectral flux and waveform periodicity from the original 16 kHz audio, pooled over the native CTC subsampling receptive field. The fourth observation is the blank log-odds from the exact stored primary posterior row. Feature extraction is causal and independent of the caller's audio chunk sizes.

`modal/student_blank_cause.py` fits a two-state diagonal-Gaussian HMM on none-labelled help-train training-speaker recordings. Quiet/speech boundaries are integrated by forward/backward EM, rather than supplied by word boundaries or a silence threshold. State 0 is oriented by its lower mean log energy. Transition rows sum to one; their self-transition probabilities represent cause duration. There are no phone-, word-, verse-, reader- or session-conditioned parameters.

The extractor requires an explicit excluded-speaker file, excludes every speaker with any non-training take, and rejects scripted mistake/repeat regions. Cached posteriors are checked against both the audio hash and primary model hash. Three speaker-disjoint partitions select the variance floor from a fixed grid using held-out training-speaker acoustic likelihood, balanced across speakers. The final model is fitted only after that selection. No practice result selects features, constants or model parameters.

The sequence scorer composes ordinary complete CTC pronunciation paths with this acoustic HMM. A quiet state emits blank with probability one. A speech state emits the unchanged primary CTC distribution, including speech-internal blank. Both candidates share exactly the same normalized state transitions, acoustic emissions and stationary start prior. The forward sum integrates phoneme placement and quiet/speech placement; it requires every target phone and preserves the CTC blank needed between identical consecutive phones.

The matched control integrates the same quiet/speech emission distributions independently against the same stationary prior at each frame. This collapses them into one blank emission and removes the state-duration dependence. It otherwise uses identical posteriors, acoustic features, complete pronunciation topology and warning windows. When the transition rows are independent, the dynamic scorer and control are mathematically equal.

A listener certificate is supplied only for a finite strict reversal: the control prefers the already-accused rival and the two-cause model prefers the complete written pronunciation. The checker exposes its original supporting pronunciation pair and exact original frame limits; the SDK cannot choose a different crop, alternative rival or partial written form. Metadata must match those strings, word and frame limits exactly. Public wrong-word confirmation becomes uncertain, while the internal confirmation and ownership ledger retain their baseline state. This preserves later candidate generation and insertion merges. Primary tokens, posteriors, tracker input, graph confirmations and existing warning thresholds are unchanged.

The Node loader supports a checksummed adjacent side-model descriptor and rejects missing or corrupt declared assets. Live audio sessions use the same scorer on available causal observations. Recorded SDK evidence is enriched after the existing gap splicing and teacher insertion enrichment, so ordinary serialization can transport the certificate to a checker replay.

## Validation and limits

Added unit checks cover exhaustive small token/state-path enumeration, complete pronunciation acceptance, repeated consonants, equality with the collapsed control under independent dynamics, candidate-order symmetry, exact-window certificates, graph preservation, normalized model validation, asset checksum/loading failures and audio chunking invariance. The probe utility also checks equality of primary evidence and tracker events and preservation of added-word decisions before/after certificate enrichment. These checks have been written but have not yet run successfully in the required remote environment.

The Python trainer passed a local syntax compilation check. `git diff --check` passed. No acoustic training, target replay, independent listening review, full practice score or bundle export has completed. No gain or preservation claim follows from the source implementation alone. An erroneous acoustic state assignment could still veto a genuine substitution; small-data state fitting and end-of-session audio retention also remain limitations.

All three supervisor requests failed before command execution with `NotFoundError: No such file or directory`, zero reported compute spend, and no returned command logs. Request 1 was training-data inspection; request 2 was a minimal `pwd` and dataset-directory listing; request 3 was the supplied six-shard build/test harness. No Node build or unit test executed. The service failure has been reported to the user through the clarification channel.

There is no trained side asset, and `.autolab/model.json` has deliberately not been declared. Running the main listener without the side model would not evaluate this experiment. Training-only model selection, target validation, complete bundle verification and (if at least three targets are removed) the authorized practice score remain required before handing this back as a completed experiment. The infrastructure error is not evidence that the acoustic premise is false, so no killed outcome or invented target count is recorded.
