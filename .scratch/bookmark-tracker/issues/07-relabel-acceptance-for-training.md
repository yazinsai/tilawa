# 07: Relabelling acceptance for training use

**What to build:** Decide, with evidence, whether proposed tlog relabels are precise enough to become distillation labels, and at which thresholds.

- **Sample:** draw a stratified sample of about 300 clips across the outcome types. Include agreements, offset relabels, multi-ayah spans, `ambiguous` and `lost` clips, and clips near the threshold boundaries.
- **Review:** a reviewer listens to each clip and records the actual span without seeing the tracker's proposal.
- **Metrics:** report relabel precision (proposed span equals the human span), the false-lock rate, the exclusion rate, and how many clips the filename label alone would have got wrong. Break these down by outcome type.
- **Thresholds:** set the adoption thresholds for fit and separation that reach the agreed precision target. Record the decision in an ADR.
- **Posteriors:** if 05 is done, compare phoneme-only and posterior-assisted results on the same reviewed sample.
- **Feedback:** verified clips go back to 02 as fixtures with `audioVerified` set.

**Blocked by:** 06.

**Status:** ready-for-human

- [ ] The reviewed sample exists, with blind human spans.
- [ ] Precision, false locks and exclusions are reported per outcome type, with uncertainty estimated by clip.
- [ ] An ADR records the adoption thresholds and the measured precision.
- [ ] Only clips passing the adopted thresholds are marked eligible for training.
