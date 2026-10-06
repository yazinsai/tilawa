# 12: Import and audit the help-page clips

**What to build:** The 902 help-page recordings (offline-tarteel `/help`, consent CC BY 4.0) turned into a labelled eval set, after a human spot-check.

**Status:** ready-for-agent

Source: `root@dokku-server:/var/lib/dokku/data/storage/offline-tarteel/help/` (`submissions.jsonl`, `audio/`). Ignore `subscribers.jsonl`.

- [ ] Audit first: `tools/help_audit.py` pulls the clips, samples them (all risky strata plus random), and serves a review page. Record the per-stratum label error rate in `docs/benchmarks/help-audit-v1.md`, and drop or fix any stratum that fails.
- [ ] Convert everything to 16 kHz mono WAV. Keep the originals.
- [ ] Expected text: join `ref` / `ayahs` on `quran.json`. Map the 1-based `text_clean` word indices (standalone waqf marks dropped) onto ref-v2 word offsets, and fail loudly on any ayah where the word counts disagree.
- [ ] Labels: planted mistakes per clip in core's event vocabulary. Repeats are non-errors. `skip_ayah` is a suspected omission; `extra_mistake` clips are dropped from scoring.
- [ ] Speaker split: assign held-out eval speakers by `speaker` before any training use. Keep the 5 heavy speakers (16–26 error clips each) in train, and stratify the split by kind.
- [ ] Dedupe: the same speaker on the same ayah under different seeds is allowed, but count it as one site for the ≥ 15-sites rule.
- [ ] Storage: audio goes in a private Modal volume. The local copy lives under `.scratch/help/` (gitignored).
