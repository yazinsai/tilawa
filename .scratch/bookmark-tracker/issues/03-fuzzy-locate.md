# 03: Fuzzy locate with learner errors and multi-ayah spans

**What to build:** Clips with learner mistakes, timing variation, or several ayahs still locate to the right span. Each result reports both how well the clip fits and how far the spoken phonemes differ from the reference.

- **Retrieval:** n-gram anchor voting, then substring Levenshtein DP constrained to word boundaries.
- **Two separate quantities:**
  - The **weighted alignment cost** is used only for ranking candidates. Its class-aware costs make confusable pairs and short vowels cheaper, and madd extension cheap.
  - The **PER** is plain, unweighted phoneme error against the matched span, computed from the same alignment path. Cheap madd in the cost must not hide differences in the PER.
- **Edit path:** the alignment/edit path for the chosen span is preserved and exposed through trace, so the checker can reuse it later.
- **Matched span:** start and end words, which may cross ayah boundaries.

**Blocked by:** 01, 02.

**Status:** ready-for-agent

- [ ] Synthetic fixtures with single and multiple substitutions, deletions and insertions locate to the correct span.
- [ ] A madd-length change lowers the alignment cost penalty but still shows up in the PER.
- [ ] A ض→د substitution locates to the correct ayah, with nonzero PER and the substitution visible in the edit path.
- [ ] Real tlog fixtures (100:6 containing 100:6 plus 100:7, 100:7 apparently containing 100:8, and 100:1 with ض→د) are added. Their expected labels are asserted only after the audio has been independently checked; until then they are regression snapshots.
- [ ] The alignment cost and PER are exposed as separate fields.
- [ ] A synthetic substitution shows in the traced alignment path (the `it.todo` carried over from 02).

## Comments

Context carried over from tickets 01 and 02. The current state is at commit `835b448`.

- The audio of the four real tlog fixtures was reviewed on 2026-10-05, and they are now `audioVerified`:
  - 303 is 100:1 with real errors (ض→د and د→ل).
  - 483 is 100:3 and already passes.
  - 816 is 100:6:1 to 100:7:3.
  - 926 is unintelligible (low volume, noise, several speakers) and **must stay `lost`**. It guards against the fuzzy matcher turning noise into a label.
- 303 and 816 are listed in `pendingFuzzyLocate` in the fixture tests and run under `it.fails`. When this ticket makes them pass, those tests will fail; move both fixtures into the normal asserted set.
- Turn the `it.todo` "shows a synthetic substitution in the traced alignment path" into a real test.
- `align.ts` already provides a unit-cost Levenshtein alignment used by trace. The weighted cost should be a separate scorer; PER must stay unweighted.
- The index asset is `packages/core/assets/ref-v1.json`. Word boundaries come from alignment, one word has an empty span, and `wordAt` handles it.
- Run tests with `corepack pnpm -C packages/core test` (global pnpm 7 is broken on Node 22). Debug with `corepack pnpm replay <fixtures>` from `packages/core`.
- The clip audio is in `.scratch/clips/`, which is gitignored. `tools/make_fixtures.py` regenerates the fixtures from `/tmp/s0.parquet`, or from `modal volume get tilawa-tlog pass1/<shard>.parquet`.
