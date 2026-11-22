# 06: Full tlog teacher pass and bucketing

**What to build:** PLAN.md's tlog processing over the whole corpus: teacher pass, relabel, buckets, speaker splits.

**Blocked by:** 03.

**Status:** done

- [x] Teacher pass over a 10% stratified shard sample, then all shards: greedy phonemes, per-phoneme confidence, and top-5 fp16 posteriors per frame. Log the cost against the $150–400 estimate.
- [x] Relabel with the threshold from ticket 03 (confirmed: 0.2).
- [x] Text-only labels for repeated text: an `ambiguous` clip whose candidates all share the same canonical phonemes (the ar-Rahman refrain, for example) gets that text as its label, without a location. See ADR 0001.
- [x] Buckets: `clean-match`, `deviant`, `ambiguous`, `reject`, with proposed cutoffs.
- [x] **Human:** tune the cutoffs on about 300 hand-checked clips (done 2026-10-06; see "Cutoff review").
- [x] Speaker clusters from WavLM-SV embeddings; splits by cluster.
- [x] Remove Quran-Lab benchmark IDs, including `tlog_holdout`.
- [x] Manifest per bucket and split, versioned (`tlog-v1`, cutoffs marked untuned).

## Cutoff review, 2026-10-06

298 of the 300 clips were answered and 2 were marked unsure. The full table is in `.scratch/v1-roadmap/bucket-review/score.md`.

- **PER from 3% to 12% is teacher noise, not recitation error.** Of the sampled deviant clips, 77 of 80 were exact recitations. Clean-match precision is flat at 96–98% across the whole grid, from PER 0 to 0.12 and teacher confidence 0.8 to 0.95.
- **Adopted** `cleanMaxPer 0.12` and `cleanMinTeacherConf 0.8`, which keeps 98.3% precision (92.8–100%). This is `tlog-v2`, with `cutoffsTuned: true`. The grid stops at 0.12, so the 0.08–0.12 band rests mostly on the 30 sampled `deviant_core` clips.

| bucket | tlog-v1 clips | tlog-v2 clips | tlog-v2 hours (train / val / test) |
|---|---|---|---|
| clean-match | 84,098 | 254,786 | 577 / 32 / 29 |
| deviant | 286,301 | 115,613 | 282 / 15 / 15 |
| ambiguous | 115,033 | 115,033 | 199 / 13 / 12 |
| reject | 234,217 | 234,217 | 604 / 29 / 31 |

- **Text-only labels:** 27 of the 30 sampled clips were exact recitations, so the refrain-style labels are safe for training.
- **Rejects:** none of the sampled reject or ambiguous clips was an exact recitation, so the reject rules hold.
- **For ticket 08:** even in `tlog-v2`, the deviant bucket is still mostly correct recitation: at PER 0.12, about 10% of deviant clips contain a mistake. Mining it for natural errors would be low-yield. Using it as weakly-labelled training data, at a lower weight, is an option for the E2 ablation.

## Results, 2026-10-06

**Corpus.** The dataset has 743 shards: 411 `clean` and 332 `unclean`, not just the 411 the ticket counted. The sample cost projected to about $20 for everything, so all 743 were run. That's 719,853 clips, of which 709,853 were decoded. The rest are 6,768 with empty or unreadable audio and 3,232 over 60 s.

**Buckets** (`tlog-v1`, proposed cutoffs):

| bucket | clips | hours | train / val / test clips |
|---|---|---|---|
| clean-match | 84,098 | 185.2 | 75,657 / 4,449 / 3,992 |
| deviant | 286,301 | 765.5 | 257,531 / 14,840 / 13,930 |
| ambiguous | 115,033 | 224.3 | 101,293 / 7,095 / 6,645 |
| reject | 234,217 | 664.6 | 209,700 / 12,751 / 11,766 |
| **total** | 719,649 | 1,839.6 | 644,181 / 39,135 / 36,333 |

Reasons within each bucket:

- clean-match: 84,156 match. 22,718 of these clean-match/deviant clips carry text-only labels.
- deviant: 283,914 have PER above 3%; 2,511 have teacher mean confidence below 0.9.
- ambiguous: 79,473 are tracker-ambiguous and have no shared text; 35,571 are locked below confidence 0.2.
- reject: 168,034 lost; 47,250 insufficient evidence; 6,989 low teacher confidence (below 0.5); 6,768 decode failed; 3,232 too long; 1,955 PER above 0.3.

**Proposed cutoffs** (`PROPOSED_CUTOFFS` in `packages/core/src/buckets/index.ts`): tracker confidence ≥ 0.2 (ADR 0001); clean-match PER ≤ 0.03 and teacher mean confidence ≥ 0.9; deviant PER ≤ 0.3; reject when teacher mean confidence < 0.5. What the data says about them, from `distributions.json`:

- Located PER has a spike at 0 (about 15% of located clips), a trough from 0.5% to 1%, and then a broad hump from 3% to 20% (median 7.5%). PLAN's ~3% therefore keeps only a quarter of located clips as clean-match. The review should show whether the 3–8% band is really recitation error or teacher and profile noise (madd length, edge drops). If it's noise, raising `cleanMaxPer` to 0.05–0.08 roughly doubles clean-match.
- Teacher mean confidence: median 0.93, p25 0.86. It barely moves clean-match: only 2,511 exact matches fall under 0.9.
- PER above 0.3 is rare (1,955 clips), because the tracker seldom locks that far off.

**Text-only labels.** Strict rule: every listed candidate has identical canonical phonemes. It recovers 22,718 clips corpus-wide, 151 of 213 ambiguous clips on shard 258 (the refrain) and 2 of 55 on shard 0.

**Benchmark removal.** All 199 `tlog_holdout` stems (`surah_ayah_id`, the same format as tlog filenames) were found in tlog. Five of them occur twice, so 204 clips were removed. The list is in `.scratch/tlog/exclude-stems.txt`, regenerated by `tools/tlog_corpus.py holdout`. Quran-Lab's other configs (`everyayah_heldout`, `qul_alnufais`) aren't tlog audio.

**Speaker clusters.** tlog has no speaker metadata; filename IDs are per recording. The method:

- Embeddings: WavLM-SV (`microsoft/wavlm-base-plus-sv`) x-vectors from each clip's first 10 s, mean-centred. Raw vectors are anisotropic: random pairs score about 0.68.
- Single-linkage on a kNN graph doesn't work: at 0.86 it chained 99% of clips into one cluster.
- What's used: spherical k-means into 35,630 micro-clusters (about 20 clips each), then average linkage over the centroids, cut at cosine distance 0.5. That gives 283 clusters; the largest holds 16.5% of clips.
- Leak diagnostic, on a probe of 50k clips: 30% have a nearest neighbour at cosine ≥ 0.95, and 5.9% of those neighbours sit in another cluster.

The threshold sweep (all files in `clusters/` on the volume):

| cut | clusters | largest | ≥0.95 neighbours in another cluster |
|---|---|---|---|
| 0.2 | 6,056 | 4.8% | 20% |
| 0.3 | 2,116 | 7.7% | 13% |
| 0.4 | 766 | 14.6% | 7.8% |
| **0.5** | 283 | 16.5% | 5.9% |
| 0.6 | 104 | 24.6% | 2.7% |

**Splits.** Splits are made by whole cluster: in a stable hash order, clusters fill test to 5%, then val to 5%. Any cluster over 1% of clips always goes to train. At cut 0.5, the 266 clusters at or under 1% hold 20% of clips, enough to fill both. Two consequences:

- Test and val are drawn only from the smaller clusters, so they may over-represent unusual voices or channels.
- About 6% of near-duplicate voices still cross the split. Treat tlog test numbers as optimistic. The held-out-speaker gate stays with the scripted-error recordings.

**Cost log.**

| item | cost |
|---|---|
| 10% sample, 76 shards (49 new; includes two launches the client killed) | $2.25 |
| projected from the sample: 743 shards at ~$0.03 each | ~$22 |
| teacher pass actual (sample + full pass + review audio extraction) | $16.78 |
| WavLM-SV embeddings, 743 shards (T4), and 3 clustering runs (L4) | $5.82 |
| **total** | **~$22.60**, against the $150–400 estimate |

Throughput was bounded by about 12 concurrent GPUs in the workspace. The full pass took about 2.5 h, sharing GPUs with the embeddings. Shards 0 and 258 were reused from ticket 03.

**Gotcha.** `modal run --detach` doesn't survive the client being killed: the signal stops the app, and in-flight shards are lost. Long runs now use `modal deploy` plus `Function.spawn`.

## How to run

Outputs:

- Volume `tilawa-tlog`: `pass1/` (743 parquet files, about 2 MB each, with top-5 posteriors), `emb/` (WavLM-SV vectors), `clusters/clusters-t{cut}-c20.parquet`, and `review/bucket-review/`.
- Local, gitignored, `.scratch/tlog/`: `pass1/`, `labels/<shard>.jsonl` (proposed labels, ref-v2, decoded-only), `clusters.tsv`, `exclude-stems.txt`, and `build/tlog-v1/`. The build holds `<bucket>/<split>.jsonl`, `manifest.json`, `clips.jsonl` and `distributions.json`.

1. **Human: tune the cutoffs.** The pack is `.scratch/v1-roadmap/bucket-review/`. It has 300 clips, seed 1, stratified as text_only 30, clean_core 40, clean_edge 50, deviant_edge 50, deviant_core 30, ambiguous 30, reject_high_per 30, reject_low_teacher 25 and reject_other 15. The audio is already extracted.

       cd packages/core && pnpm buckets serve --dir ../../.scratch/v1-roadmap/bucket-review

   Open http://localhost:5178. Each clip plays with the text it would be trained on. Keys: `1` exact, `2` right passage but with mistakes, `3` wrong passage, `4` no usable recitation, `u` unsure. Also `space` play/pause, `r` replay, `←`/`→` move, `j` next unanswered, `n` add a note. Answers are appended to `answers.jsonl`, and the latest answer for a clip wins. Expect about 1 hour.

   If the audio is missing:

       modal volume get tilawa-tlog review/bucket-review/ /tmp/br && mv /tmp/br/bucket-review/*.flac .scratch/v1-roadmap/bucket-review/audio/

2. Score. Answers are weighted back to each stratum's population. This writes `score.md` with verdicts per stratum and a grid over clean max PER (0–0.12) × teacher confidence (0.8–0.95). The grid shows clean precision with its CI, clean share, and the deviant right-passage rate. It also prints the largest clean-match whose precision is at or above the target.

       pnpm buckets score --dir ../../.scratch/v1-roadmap/bucket-review --target 0.95

3. Rebuild with the tuned cutoffs, as a new version, then set `cutoffsTuned` in its `manifest.json`:

       echo '{"cleanMaxPer": 0.05, "cleanMinTeacherConf": 0.9}' > ../../.scratch/v1-roadmap/bucket-review/cutoffs.json
       NODE_OPTIONS=--max-old-space-size=16000 pnpm buckets build --labels ../../.scratch/tlog/labels --out ../../.scratch/tlog/build --version tlog-v2 --exclude ../../.scratch/tlog/exclude-stems.txt --clusters ../../.scratch/tlog/clusters.tsv --cutoffs ../../.scratch/v1-roadmap/bucket-review/cutoffs.json

To rebuild everything from scratch:

    modal deploy modal/tlog_pass.py
    python -c "import modal; modal.Function.from_name('tilawa-tlog-pass', 'run_all').spawn(1, 0, 'clean,unclean')"   # 10, 0 for the 10% sample
    modal deploy modal/speaker_embed.py
    python -c "import modal; modal.Function.from_name('tilawa-speaker-embed', 'run_all').spawn()"
    python -c "import modal; modal.Function.from_name('tilawa-speaker-embed', 'cluster').spawn('0.5', 20)"
    T="uv run --python cpython-3.12-macos-aarch64-none tools/tlog_corpus.py"
    $T pull --dir .scratch/tlog && $T relabel --dir .scratch/tlog --jobs 8      # about 30 min for 743 shards
    $T holdout --dir .scratch/tlog
    $T clusters --dir .scratch/tlog --file clusters/clusters-t0.5-c20.parquet
    cd packages/core && NODE_OPTIONS=--max-old-space-size=16000 pnpm buckets build --labels ../../.scratch/tlog/labels --out ../../.scratch/tlog/build --version tlog-v1 --exclude ../../.scratch/tlog/exclude-stems.txt --clusters ../../.scratch/tlog/clusters.tsv
    pnpm buckets sample --clips ../../.scratch/tlog/build/tlog-v1/clips.jsonl --dir ../../.scratch/v1-roadmap/bucket-review
    cd ../.. && modal run modal/tlog_pass.py::extract --rows-tsv .scratch/v1-roadmap/bucket-review/rows.tsv --dest review/bucket-review

Code: `packages/core/src/buckets/` (bucketing, text-only labels, manifests, the cutoff review), `modal/speaker_embed.py`, `tools/tlog_corpus.py`. Tests are in `packages/core/test/buckets.test.ts` and `bucket-review.test.ts`.
