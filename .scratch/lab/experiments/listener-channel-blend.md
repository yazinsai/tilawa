# Conservative channel adaptation

Status: registered; no scores observed. Not folded.

The full 200-update microphone adaptation reduced phoneme errors and missed mistakes, but introduced ten phone false-warning incidents. It fails the acceptance rule. This separate experiment tests one fixed 75% shipped / 25% adapted weight interpolation, preserving the shipped architecture, greedy decoder and checker. The coefficient was fixed before inspecting any blended output and will not be swept or adjusted against practice results. No new training or benchmark labels are used. Integer buffers retain their shipped values. Export and quantization follow the same path as the shipped listener.

Parents: `spliced-clean` and `channel-adapt-v1`; destination `channel-blend25-v1`; listener `channel-blend25-int8`. Parent checkpoint hashes and the interpolated checkpoint hash are recorded in the model volume's `blend.json`.

Prediction registered before building: three fewer listener-attributed and overall wrong calls. User gate: fewer missed mistakes, zero new false-alarm incidents, gains across several chapters. Report individual new misses as well. Evaluate both phone and studio copies of the unchanged 100-session practice set. No exam evidence is decoded or inspected.
