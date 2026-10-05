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

## Comments

- Review tooling: `packages/core/src/review/` (`pnpm review sample|serve|score`), audio via `tools/extract_review_audio.py`. Data in `.scratch/bookmark-tracker/review/`: `proposals/` (shard 0, decoded and posteriors, current tracker), `sample.json` (302 clips, seed 1), `answers.jsonl` (appended by the page; latest answer per clip wins). Audio is gitignored; re-extract with:
  `uv run --python cpython-3.12-macos-aarch64-none tools/extract_review_audio.py --dir .scratch/bookmark-tracker/review --shard clean-00000-of-00411`
- Review: `cd packages/core && pnpm review serve --dir ../../.scratch/bookmark-tracker/review`, open http://localhost:5177. The page shows only the surah text and the audio: no filename ayah, no tracker output.
- Score: `pnpm review score --dir ../../.scratch/bookmark-tracker/review --labels ../../.scratch/bookmark-tracker/review/proposals/proposed-labels.jsonl --posterior-labels ../../.scratch/bookmark-tracker/review/proposals/proposed-labels.posteriors.jsonl --target 0.98` writes `score.md` / `score.json`.
- Strata are assigned in order: near threshold (confidence 0.04–0.2), decoded and posterior runs disagree, then outcome (agrees, part of ayah, offset, multi-ayah, other surah, ambiguous, lost, insufficient). Run-level numbers are weighted by each stratum's share of the 966 decoded clips; CIs are Wilson per stratum, Agresti–Coull for weighted rates. The threshold pick uses the CI's lower bound, not the point estimate.
- Still to do after review: the ADR, eligible-clip export at the adopted threshold, and turning verified clips into `audioVerified` fixtures.
