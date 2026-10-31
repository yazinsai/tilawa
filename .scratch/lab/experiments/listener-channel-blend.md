# Conservative channel adaptation

Status: registered; no scores observed. Not folded.

The full 200-update microphone adaptation reduced phoneme errors and missed mistakes, but introduced ten phone false-warning incidents. It fails the acceptance rule. This separate experiment tests one fixed 75% shipped / 25% adapted weight interpolation, preserving the shipped architecture, greedy decoder and checker. The coefficient was fixed before inspecting any blended output and will not be swept or adjusted against practice results. No new training or benchmark labels are used. Integer buffers retain their shipped values. Export and quantization follow the same path as the shipped listener.

Parents: `spliced-clean` and `channel-adapt-v1`; destination `channel-blend25-v1`; listener `channel-blend25-int8`. Parent checkpoint hashes and the interpolated checkpoint hash are recorded in the model volume's `blend.json`.

Prediction registered before building: three fewer listener-attributed and overall wrong calls. User gate: fewer missed mistakes, zero new false-alarm incidents, gains across several chapters. Report individual new misses as well. Evaluate both phone and studio copies of the unchanged 100-session practice set. No exam evidence is decoded or inspected.

## Composition fixed before measurement

Include the exact checker source and addition tests measured in `checker-extra-segmentation` (`b2fe98d`; final record `4001bd6`). Its independent practice result is 64 misses and 25 warnings on phone, 45 misses and 13 warnings on studio. Compare this combined candidate against both the shipped baseline and that checker-only result; listener improvement must exceed the known checker-only gains and add no false warnings. This tests their interaction before proposing a fold. The registered -3/-3 prediction is unchanged; it is not evidence for an isolated listener effect. No blended practice outputs have been inspected.

Blend app `ap-AvT7l0yS4XygRpZ3RNSRuE` validated all 829 tensors for matching shape/type and finite values. Checkpoint SHA256 `c99ffcf318bfbd2748961572870fd5aa6e98b93abbaba3b5d1ad38ed86648f0e`.

Export app `ap-NLiR1e2oE7RGDmkIoZ3gK4`: six fixed help clips, zero collapsed transcript edits against PyTorch, exact frame-length match. As with the parent model, probability parity is not exact (maximum absolute difference 0.615); evaluation uses the actual final int8 artifact. Quantization app `ap-0QoA881TfJWsvLajghkxQH`: 67,759,744 bytes, SHA256 `7231af534451698706f7a5f81087aaf74cb653a9b37c564a9bfc340cfd118d9e`. The same model architecture and runtime apply.
