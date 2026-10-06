# 12: Import and audit the help-page clips

**What to build:** The 902 help-page recordings (offline-tarteel `/help`, consent CC BY 4.0) turned into a labelled eval set, after a human spot-check.

**Status:** ready-for-agent

Source: `root@dokku-server:/var/lib/dokku/data/storage/offline-tarteel/help/` (`submissions.jsonl`, `audio/`). Ignore `subscribers.jsonl`.

- [ ] Audit first: `tools/help_audit.py` pulls the clips, samples them (all risky strata plus random), and serves a review page. Record the per-stratum label error rate in `docs/benchmarks/help-audit-v1.md`, and drop or fix any stratum that fails.
- [ ] Wrong-passage clips: `modal/help_pass.py` + `help_audit.py check` flag 20 clips whose teacher decode fits another passage (≤ 25% PER and ≥ 15 points better than expected), plus 56 with poor fit to anything.
  - Root cause for most of them: 20 of 27 wrong v1 clips match the **v2** prompt regenerated from the same seed. The screen showed v2, but the server stored the v1 label. `.scratch/help/version_swap.json` holds the v2 labels. Relabel them after the audit confirms them by ear, rather than dropping them.
  - Stored labels are otherwise faithful: all 583 v1/v2 records regenerate exactly from their seeds.
  - 10 parallel-ayah substitutions (79:39 and 79:41) legitimately decode closer to the borrowed ayah, and are excluded from the flag.
- [ ] Convert everything to 16 kHz mono WAV. Keep the originals.
- [ ] Expected text: join `ref` / `ayahs` on `quran.json`. Map the 1-based `text_clean` word indices (standalone waqf marks dropped) onto ref-v2 word offsets, and fail loudly on any ayah where the word counts disagree.
- [ ] Labels: planted mistakes per clip in core's event vocabulary. Repeats are non-errors. `skip_ayah` is a suspected omission; `extra_mistake` clips are dropped from scoring.
- [ ] Speaker split: assign held-out eval speakers by `speaker` before any training use. Keep the 5 heavy speakers (16–26 error clips each) in train, and stratify the split by kind.
- [ ] Dedupe: the same speaker on the same ayah under different seeds is allowed, but count it as one site for the ≥ 15-sites rule.
- [ ] Storage: audio goes in a private Modal volume. The local copy lives under `.scratch/help/` (gitignored).
