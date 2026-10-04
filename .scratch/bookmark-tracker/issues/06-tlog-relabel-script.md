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

**Status:** ready-for-agent

- [ ] The script runs on the existing smoke-test shard and produces the per-clip output and summary.
- [ ] Every excluded clip has exactly one exclusion reason.
- [ ] Repeated runs on the same input give identical output.
- [ ] The output is clearly marked as proposed labels, not training labels. Running on the smoke shard verifies the script only, not relabelling precision.
