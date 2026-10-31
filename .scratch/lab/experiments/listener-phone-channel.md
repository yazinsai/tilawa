# Complete phone-channel training

Status: registered; training complete; export and practice evaluation pending. Not folded.

The bandpass adaptation and fixed 25% interpolation both fail the zero-new-false-alarm gate. Static inspection shows a training mismatch: only 20% room-response and 50% noise augmentation, noise scaled to whole-clip RMS, while every phone copy uses room response, a 120–7000 Hz band, and noise at 8–22 dB against active speech RMS.

Fixed candidate: initialize from shipped `spliced-clean`; same 50-hour gold-excluded training subset, seed 0, 200 updates, one epoch, learning rate 0.0015, batch budget 800 seconds, spliced CTC and KD weight 2. Apply the existing pure `lab_sessions.degrade` channel function to 75% of training clips, using only existing training clips and augmentation-bank noise/RIR. The other 25% retain the original augmentation recipe. Randomize noise offset and draw SNR uniformly from 8–22 dB. Keep gain/spec augmentation. No benchmark speech, labels, or per-case settings enter training. Choose the final scheduled checkpoint, no checkpoint or parameter sweep.

Include the exact passing checker fix. Compare with both shipped main (67 misses, 25 alarms) and checker-only (64 misses, 25 alarms), phone and studio, all 100 practice sessions. Require fewer misses, zero new false-alarm incidents and gains in multiple chapters; disclose individual newly missed cases. Expected reductions registered before code: five listener-attributed, six overall. No exam data.

Pre-training validation: local Python compilation passes. Modal `ap-6i0zLNRPjOIEknhdPmf9oM` verifies unchanged input, deterministic outputs, finite float32 output of identical length and measured active-speech SNR 15.000000013 dB for a 15 dB request.

Training app `ap-bneuNqx2LhpZr65rtdfluP` completed exactly 200 updates from source step 30287, with 21,161 training clips and 50.0008 hours, matching the earlier adaptation's sampled data. No resume occurred. The final checkpoint is the scheduled 200-step checkpoint, not one selected by scores. The 300-clip clean diagnostic is recorded on the volume and is not a model-selection criterion.

Export app `ap-SLtbUlheW0oTsL4Zqxzxso`: all six fixed help fixtures have zero transcript edits against PyTorch and identical frame lengths. Posterior probabilities are not identical (maximum difference 0.597), so the final int8 artifact will be evaluated directly rather than inferred from PyTorch diagnostics.

Int8 publication app `ap-NdxCH4xBzbmSa5VuYzDWGx`: 67,759,744 bytes; SHA256 `00cfec2268dd521689645399290ada8ea093a46b902f9fb8bcd89e585fb7cee5`. Model directory: `tilawa-lab:/data/models/phone-channel-int8`.

## Recorded comparison points

All rows use the same lab-v1.5 100-session practice set; phone is the headline. Incident counts, not only totals, govern acceptance.

| Candidate | Misses | False alarms | New phone alarms | Status |
| --- | ---: | ---: | ---: | --- |
| Shipped main | 67 | 25 | baseline | baseline |
| Checker only | 64 | 25 | 0 | passes; three recoveries across chapters 2, 9 and 35 |
| Audio-only beam decoder | 72 | 25 | 7 vs main | fails |
| Bandpass adaptation | 63 | 30 | 10 vs main | fails |
| Fixed 25% weight blend + checker | 65 | 26 | 4 vs checker only | fails |
| Complete phone-channel training + checker | pending | pending | pending | pending |

The checker-only result also has no new missed cases or studio alarms and passes all 607 core tests. Its oracle arm improves from 50 misses + 6 false alarms to 46 + 5, with no new oracle alarm incidents. It remains ready for the user's fold decision regardless of this listener experiment.

`decode-phone-channel-int8-0cfd306-20261031T150716Z`: 200/200 SDK jobs succeeded on the 100 practice sessions, phone and studio. Model and evidence hashes are stored in the decode record. No exam files were generated.
