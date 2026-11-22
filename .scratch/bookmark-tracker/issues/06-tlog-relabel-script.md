# 06: tlog relabelling script (dry run)

**What to build:** A Node script runs the phoneme-only tracker over the tlog `pass1` output, with the filename ayah as a prior. For every clip it writes a proposed label, without changing anything in the training data.

- **Output per clip:**
  - the original filename label;
  - the proposed span, or none;
  - the tracker state;
  - fit, separation, alignment cost and PER;
  - the competing candidates;
  - an exclusion reason: `ambiguous`, `lost`, `insufficient_evidence`, `decode_failed`, or `too_long`.
- **Summary:** counts by outcome, agreement with the filename label, and the distribution of offset patterns (for example "off by +1 ayah" or "contains 2 ayahs").
- **Posteriors:** an option re-runs with posterior reranking once 05 is done, so the two runs can be compared.

**Blocked by:** 04.

**Status:** done

- [x] The script runs on the existing smoke-test shard and produces the per-clip output and summary.
- [x] Every excluded clip has exactly one exclusion reason.
- [x] Repeated runs on the same input give identical output.
- [x] The output is clearly marked as proposed labels, not training labels. Running on the smoke shard verifies the script only, not relabelling precision.

## Comments

- Export: `uv run --python cpython-3.12-macos-aarch64-none tools/export_pass1.py --parquet /tmp/s0.parquet --out /tmp/s0.pass1.jsonl` (Node can't read parquet with zero deps). Evidence uses the greedy decode of `top5_ids`, which carries frame spans; on shard 0 it differs from the stored `hyp` in 12 of 970 rows (near-ties broken differently by `max` and `topk`), so shard 0 shows 154 lost here vs 153 in 04.
- Run: `pnpm --filter @tilawa/core relabel /tmp/s0.pass1.jsonl --out /tmp/relabel-s0 [--posteriors] [--prior-weight 1]`. Writes `proposed-labels[.posteriors].jsonl` and `proposed-summary[.posteriors].json`; every record and the summary carry `labelKind: "proposed"`. About 45 s per shard.
- Code in `packages/core/src/relabel/`, tests in `test/relabel.test.ts`. The tracker trace now carries `decision` (fit, separation, lead and required edits for the top fitting candidate), which the script reports.
- Exclusions: `acquiring` and an empty decode map to `insufficient_evidence`; `decode_failed`/`too_long` come from pass1 `status`.
- Agreement = locked span lies within the filename ayah. Candidates are distinct shortlist locations ranked by cost without the prior.
- Shard 0 (smoke check only, not relabelling precision): 696 locked, 154 lost, 64 insufficient, 56 ambiguous. Of locked, 464 (67%) agree. Top offsets: 50 off by −1, 50 contain previous+filename ayah, 45 contain filename+next, 21 other surah. Two runs are byte-identical.
- `--posteriors` currently gives identical output, because the tracker ignores posteriors until 05 lands. Diff the two jsonl files then.
