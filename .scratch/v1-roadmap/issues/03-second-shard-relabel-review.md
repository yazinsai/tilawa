# 03: Second-shard relabel review

**What to build:** A blind review of a second tlog shard on a long surah with repeated phrases, re-checking ADR 0001's precision outside surah 100.

**Status:** done

- [x] Choose a shard filename-labelled ar-Rahman (preferred, because of the refrain) or al-Baqarah.
- [x] Run the relabel and draw a stratified sample of about 150 clips, weighted toward low-confidence and posterior-disagreement strata.
- [x] Review blind with the existing page and score at ±1 word.
- [x] Update ADR 0001 with the second shard: confirm confidence ≥ 0.2 or move the threshold.

## How to run

Prepared: shard `clean-00258-of-00411` (970 clips, all filename-labelled 55 ar-Rahman; 124 have empty audio in the dataset, so 846 decoded). Proposals for the decoded and posterior runs are in `.scratch/v1-roadmap/review-55/proposals/`. The sample is in `sample.json`: 150 clips, seed 1. Its strata are near_threshold 50/68, posteriors_disagree 10/10 (all of them), agrees 15/144, part_of_ayah 8/8, offset 15/77, multi_ayah 15/122, other_surah 10/29, ambiguous 15/178, lost 8/181, insufficient 4/26.

Audio is gitignored. If it's missing, re-extract it:

    uv run --python cpython-3.12-macos-aarch64-none tools/extract_review_audio.py --dir .scratch/v1-roadmap/review-55 --shard clean-00258-of-00411

1. Review, then open http://localhost:5177. Answers are appended to `answers.jsonl`.

       cd packages/core && pnpm review serve --dir ../../.scratch/v1-roadmap/review-55

   Use `G` to type a span such as `34`, `13-14` or `13:2-14:3`, because the ayah keys only reach 12.

2. Score:

       pnpm review score --dir ../../.scratch/v1-roadmap/review-55 --labels ../../.scratch/v1-roadmap/review-55/proposals/proposed-labels.jsonl --posterior-labels ../../.scratch/v1-roadmap/review-55/proposals/proposed-labels.posteriors.jsonl --target 0.98

3. Eligible, at the threshold the ADR ends up with:

       pnpm review eligible --dir ../../.scratch/v1-roadmap/review-55 --labels ../../.scratch/v1-roadmap/review-55/proposals/proposed-labels.jsonl --min-confidence 0.2

To rebuild from scratch:

    modal run modal/tlog_pass.py --shard clean-00258-of-00411
    modal volume get tilawa-tlog pass1/clean-00258-of-00411.parquet /tmp/s258.parquet
    uv run --python cpython-3.12-macos-aarch64-none tools/export_pass1.py --parquet /tmp/s258.parquet --out /tmp/s258.pass1.jsonl
    pnpm --filter @tilawa/core relabel /tmp/s258.pass1.jsonl --out $PWD/.scratch/v1-roadmap/review-55/proposals [--posteriors]
    cd packages/core && pnpm review sample --labels ... --posterior-labels ... --dir ../../.scratch/v1-roadmap/review-55 --quota near_threshold=50,posteriors_disagree=30,agrees=15,part_of_ayah=8,offset=15,multi_ayah=15,other_surah=10,ambiguous=15,lost=8,insufficient=4

**Result, 2026-10-06.** 0.2 confirmed: 97.9% precision on the decoded run, 45% of the shard kept. Refrain-only clips are never locked. See the second-shard section of ADR 0001.
