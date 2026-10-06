# 12: Import and audit the help-page clips

**What to build:** The 902 help-page recordings (offline-tarteel `/help`, consent CC BY 4.0) turned into a labelled eval set, after a human spot-check.

**Status:** ready-for-human

Source: `root@dokku-server:/var/lib/dokku/data/storage/offline-tarteel/help/` (`submissions.jsonl`, `audio/`). Ignore `subscribers.jsonl`.

- [ ] Audit first: `tools/help_audit.py` pulls the clips, samples them (all risky strata plus random), and serves a review page. Record the per-stratum label error rate in `docs/benchmarks/help-audit-v1.md`, and drop or fix any stratum that fails.
- [ ] Wrong-passage clips: `modal/help_pass.py` + `help_audit.py check` flag 20 clips whose teacher decode fits another passage (≤ 25% PER and ≥ 15 points better than expected), plus 56 with poor fit to anything.
  - Root cause for most of them: 20 of 27 wrong v1 clips match the **v2** prompt regenerated from the same seed. The screen showed v2, but the server stored the v1 label. `.scratch/help/version_swap.json` holds the v2 labels. Relabel them after the audit confirms them by ear, rather than dropping them.
  - Stored labels are otherwise faithful: all 583 v1/v2 records regenerate exactly from their seeds.
  - 10 parallel-ayah substitutions (79:39 and 79:41) legitimately decode closer to the borrowed ayah, and are excluded from the flag.
- [x] Convert everything to 16 kHz mono WAV. Keep the originals.
- [x] Expected text: join `ref` / `ayahs` on `quran.json`. Map the 1-based `text_clean` word indices (standalone waqf marks dropped) onto ref-v2 word offsets, and fail loudly on any ayah where the word counts disagree.
- [x] Labels: planted mistakes per clip in core's event vocabulary. Repeats are non-errors. `skip_ayah` is a suspected omission; `extra_mistake` clips are dropped from scoring.
- [x] Speaker split: assign held-out eval speakers by `speaker` before any training use. Keep the 5 heavy speakers (16–26 error clips each) in train, and stratify the split by kind.
- [x] Dedupe: the same speaker on the same ayah under different seeds is allowed, but count it as one site for the ≥ 15-sites rule.
- [x] Storage: audio goes in a private Modal volume. The local copy lives under `.scratch/help/` (gitignored).

## How to run

```sh
uv run --with pytest pytest tools/test_help_import.py
uv run tools/help_import.py build            # .scratch/help/labels.jsonl, prints the split and site counts
uv run tools/help_import.py build --swap     # only once the audit confirms the version_swap.json clips by ear
uv run tools/help_import.py upload           # wav/ + labels.jsonl -> Modal volume tilawa-help
```

Set `HELP_DIR` when running from a worktree, since `.scratch/help/` lives in the main checkout.

## Comments

- **Left for the audit:** listening, the per-stratum error rates in `help-audit-v1.md`, and confirming the 20 version swaps by ear. After that, rerun `build --swap` and `upload`. `--swap` is opt-in and marks each relabelled clip `"relabel": "version_swap"`. The split is drawn from the stored labels, so the swap never moves a speaker.
- **Conversion:** `help_audit.py pull` already converted the clips. `build` checks that every `wav/` file is 16 kHz mono and reconverts any that aren't. The originals stay in `raw/audio`.
- **Word mapping:** quran.json prepends the basmala to the first ayah of every surah except 1:1 and 9:1, and the help page counts those 4 words. So those ayahs map with a shift of 4. Any other word-count disagreement stops the build. 22 clip ayahs need the shift. 115 ayahs disagree overall, for example the split of بعدما in 2:181, but no clip uses the others.
- **Excluded clip:** `muo3pp0v_f7928cdd` (v1) plants a repeat on الرحمن in the basmala before 104:1. That word has no ref-v2 location, so the clip is written as unscored with an `excluded` reason. It's the only one.
- **Labels (`labels.jsonl`):** each clip gets `events` holding the final checker state a correct engine should reach, with a 1-based ref-v2 `WordRef`.
  - substitution → `wrong_word`, skip_word → `omitted_word`, vowel → `short_vowel`. These are `confirmed`.
  - tajweed no_shadda → `missing_shadda`, madd_short → `madd_natural`, light_emphatic (all 15 are ص) → `consonant_sad_sin`.
  - no_ghunna (19 clips) has no core class, so it goes in `unsupported`.
  - skip_ayah → `suspected`. It is `skipped_passage` when the ayah has 3 or more words (the checker's threshold) and `omitted_word` otherwise.
  - Repeats go in `non_errors` with no event. `extra_mistake` clips (37) have `scored: false`.
- **Split:** 135 speakers. The 5 heavy speakers, those with 16 or more error clips, are in train. The others are dealt greedily (seed 12) so each kind's eval share is near 50% of the non-heavy clips. Result: 30 eval speakers with 413 clips, and 105 train speakers with 489 clips.
- **Sites** (speaker × ayah, scored only):
  - eval: wrong_word 33, omitted_word 36, skipped_passage 17, short_vowel 30, consonant_sad_sin 4, madd_natural 1, missing_shadda 0.
  - train: wrong_word 45, omitted_word 61, skipped_passage 27, short_vowel 39, consonant_sad_sin 7, madd_natural 9, missing_shadda 5.
  - Only the memorization classes and short_vowel clear 15 sites in eval. The rare tajweed classes can't be scored from this set.
- **Storage:** private Modal volume `tilawa-help` holds `/wav` (902 files) and `/labels.jsonl`. The labels uploaded there are the unswapped ones.

