# Microphone-band adaptation

Status: registered; training checks pass; practice evaluation pending.

## Fixed plan

Start from `spliced-clean`, preserve the model architecture, greedy decoder, spliced CTC target and KD weight 2. Use up to 50 hours from the existing gold-excluded `clean-match` training split, seed 0, one epoch capped at 200 optimizer steps, learning rate 0.0015, batch budget 800 seconds. No lab practice or exam audio enters training. Use the final scheduled checkpoint, not a checkpoint chosen from practice scores.

Apply a training-only random fourth-order bandpass to 75% of training clips, with high-pass 60–180 Hz and low-pass 4500–7600 Hz. It follows the existing room response and precedes existing noise augmentation, and uses zero-phase filtering so phoneme timing targets stay aligned. The remaining 25% preserve the original channel treatment. Existing RIR, noise, gain and spectral augmentation settings remain. Random augmentation is fixed per training clip for this one-epoch pilot.

This is a bounded adaptation candidate, not a causal ablation against unfiltered fine-tuning. The hypothesis is motivated by 17 of 19 listener-attributed false-warning incidents disappearing on the studio copy. The earlier acoustic prefix-decoder experiment regressed and is not used here.

The card predicted five fewer listener-attributed and overall wrong calls before implementation. Accept only fewer missed mistakes, zero new false-warning incidents and no new missed cases across the full practice set. Assess gains across chapters and old/new/clean slices. Decoder and checker stay fixed.

## Implementation and checks

- The trainer can initialize a new run from a complete existing student checkpoint without inheriting its optimizer schedule. It retains the pretrained model's internal step offset and persists it for resumption.
- A max-step cap bounds this pilot; existing training defaults remain unchanged.
- The exact training key list is saved beside the checkpoint; the existing loader excludes gold training rows.
- `channel_checks` on Modal passed: waveform length/type and original samples preserved; fixed-seed output reproducible; very low and high tones attenuated, 1 kHz preserved.
