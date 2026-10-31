# Complete phone-channel training

Status: registered; channel checks pass; training pending. Not folded.

The bandpass adaptation and fixed 25% interpolation both fail the zero-new-false-alarm gate. Static inspection shows a training mismatch: only 20% room-response and 50% noise augmentation, noise scaled to whole-clip RMS, while every phone copy uses room response, a 120–7000 Hz band, and noise at 8–22 dB against active speech RMS.

Fixed candidate: initialize from shipped `spliced-clean`; same 50-hour gold-excluded training subset, seed 0, 200 updates, one epoch, learning rate 0.0015, batch budget 800 seconds, spliced CTC and KD weight 2. Apply the existing pure `lab_sessions.degrade` channel function to 75% of training clips, using only existing training clips and augmentation-bank noise/RIR. The other 25% retain the original augmentation recipe. Randomize noise offset and draw SNR uniformly from 8–22 dB. Keep gain/spec augmentation. No benchmark speech, labels, or per-case settings enter training. Choose the final scheduled checkpoint, no checkpoint or parameter sweep.

Include the exact passing checker fix. Compare with both shipped main (67 misses, 25 alarms) and checker-only (64 misses, 25 alarms), phone and studio, all 100 practice sessions. Require fewer misses, zero new false-alarm incidents and gains in multiple chapters; disclose individual newly missed cases. Expected reductions registered before code: five listener-attributed, six overall. No exam data.

Pre-training validation: local Python compilation passes. Modal `ap-6i0zLNRPjOIEknhdPmf9oM` verifies unchanged input, deterministic outputs, finite float32 output of identical length and measured active-speech SNR 15.000000013 dB for a 15 dB request.
