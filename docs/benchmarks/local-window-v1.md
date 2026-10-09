# Local-window warning confirmation v1 (ticket 07)

Commit adbcafe (scoreboard rows @ 77f4594, after merging v1-roadmap; same counts). Before the checker confirms a wrong, omitted or added word, it scores the chunk posteriors over the error's word plus one recited word either side, from the first to the last decoded phoneme aligned there. It compares two CTC likelihoods, each summed over all timings (`PosteriorFrames.nll`): the written words, and what was decoded (the observed phonemes in place of the word; the word dropped for an omission; the inserted phonemes added for an added word). If the decode does not beat the written words by at least `SUPPORT` = 10 nats, the error ends `uncertain` instead of `confirmed`. The transcript, alignment, self-correction, repeat, skip and confirm-delay rules are unchanged. Skipped passages are not checked. Without posteriors the check does nothing.

The check is on by default (`CheckerPolicy.localWindow`, `false` turns it off) in both arms. The SDK session now asks the listener for top-5 posteriors and passes them to the checker only. The tracker still gets the decode alone, so tracker metrics are unchanged.

## Protocol

1. **Dev pilot, train split only.** The cached teacher evidence covers only the eval split. `tools/decode_cpu.py` decoded all 433 scored train clips (98 speakers, none in the acceptance split) with the same Muaalem model on local CPU in fp32. A check on 10 eval clips against the GPU bf16 evidence found a 0.1% phoneme difference. The pilot used every train clip, not a 100-clip sample. That gives 32 false-warning candidates (fewer than the proposed 40, so all were used), 81 confirmed real mistakes, 211 clean clips and 30 repeat clips.
2. **Frozen on dev.** The threshold sweep in the declared arm, as hits / false positives / misses:

   | nats | off | 0 | 3 | 6 | 8 | **10** | 12 | 16 | 20 | 30 | 40 |
   |---|---|---|---|---|---|---|---|---|---|---|---|
   | dev declared | 81/32/8 | 81/28/8 | 81/27/8 | 81/24/8 | 81/20/8 | **81/19/8** | 81/19/8 | 80/16/9 | 79/14/10 | 78/11/11 | 73/9/16 |

   10 nats sits in the 8–12 plateau, before the first lost hit at 16. The hits lost above that are omissions. Dev free arm: 50/13/39 → 50/11/39.
3. **One held-out replay** on the 388 scored eval clips (29 speakers), with 10 frozen, enabled vs disabled. Nothing was retuned afterwards.
4. **Shipped student, separately:** `spliced-clean` int8 (`full`) on onnxruntime-node through the SDK session (20 ms pushes, 40 ms evidence). The cached `.scratch/sdk-help/spliced-clean-int8-node` evidence had no posteriors, so it was regenerated with them into `spliced-clean-int8-node-lw`. The decode is identical to the cached run (294/388 identical to PyTorch, as before), and the SDK and the core replay give the same counts.

## Results, held-out (hits / false positives / misses)

| Listener | Arm | Disabled | Enabled | Precision | Recall |
|---|---|---|---|---|---|
| Teacher | Declared | 57 / 17 / 7 | **57 / 11 / 7** | 77.0% → 83.8% | 89.1% → 89.1% |
| Teacher | Free | 33 / 10 / 31 | **33 / 6 / 31** | 76.7% → 84.6% | 51.6% → 51.6% |
| int8 `full` | Declared | 56 / 12 / 8 | **56 / 8 / 8** | 82.4% → 87.5% | 87.5% → 87.5% |
| int8 `full` | Free | 28 / 8 / 36 | **28 / 4 / 36** | 77.8% → 87.5% | 43.8% → 43.8% |

By class and slice, disabled → enabled:

| | Teacher declared | Teacher free | int8 declared | int8 free |
|---|---|---|---|---|
| wrong_word hits / FP / misses | 24/12/5 → 24/6/5 | 2/6/27 → 2/2/27 | 24/8/5 → 24/5/5 | 2/5/27 → 2/2/27 |
| omitted_word | 33/2/2 → 33/2/2 | 31/1/4 → 31/1/4 | 32/1/3 → 32/1/3 | 26/0/9 → 26/0/9 |
| added_word FP | 1 → 1 | 1 → 1 | 1 → 0 | 1 → 0 |
| skipped_passage FP | 2 → 2 | 2 → 2 | 2 → 2 | 2 → 2 |
| Clean-clip warnings (225 clips) | 6 → 5 | 2 → 2 | 6 → 4 | 4 → 2 |
| FP on repeats (23 clips) | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| FP on a passage's first or last word | 3 → 2 | 0 → 0 | 1 → 1 | 0 → 0 |
| Warnings on vowel clips (30 clips) | 2 → 1 | 2 → 1 | 0 → 0 | 0 → 0 |
| Newly missed real mistakes | 0 | 0 | 0 | 0 |

**Proposed intermediate criterion (teacher declared, FP ≤ 12 with ≥ 55/64 hits): met,** at 57 hits and 11 false positives. The 90% shipping gate is not met: that needs about 6 false positives at 55 hits.

The six teacher declared-arm false positives removed are all decoder noise on an otherwise correct word: إيّاك → قيّات (1:5:1), تعبدون → تعمضو (109:2:4), عابدٌ → لاادل (109:4:3), نعبد → نعضم (1:5:2, a repeat clip), المغضوب → المعضوم (1:7:6, a vowel clip), ثمّ → ثوو (79:22:1, clean). Eleven remain. Five are clean-clip wrong words where the decode clearly differs, such as الجحيم → لهجڇم and لِيكفّر → ليغفر at a clip edge. The others are 2 skipped passages and 2 omissions (not checked or not vetoed), 1 added word and 1 wrong word next to a planted error.

**Vowel sites.** The check only changes the confirmation of memorization warnings. It never edits the decode, so the 30 planted short-vowel positives at 27 sites keep the same evidence: teacher E1 greedy 19/30 and posterior 28/30, unchanged by construction. On vowel clips, the one memorization warning removed was the decoder-noise المغضوب above. The vowel errors themselves are untouched.

## Cost

Shipped int8 runtime on this laptop (onnxruntime-node): RTF went from 0.0494 to 0.0506 with top-5 posteriors on and the window check. Highlight p95 was 490 ms and suspected p95 1013 ms (`sdk-help-spliced-clean-int8-lw-v1.json`). The check runs only when an error is about to be confirmed. Its cost is O(window frames × 2 × window phonemes), a few thousand operations. It adds no wait: confirmation still triggers on the existing delay or next-word onset. The device measurement (Android floor, iPhone XR/11) and the 30-minute clean long-session check are not done.

## Limits

- Top-5 truncation: unlisted tokens share the leftover probability mass (`PosteriorFrames.row`). A written phoneme missing from the top 5 therefore scores low, which favours keeping the warning. A veto never relies on mass that isn't listed.
- The decoded alternative comes from the greedy decode, so it is always listed. Other plausible substitutions are not enumerated.
- The window is always the word plus one word either side. It doesn't expand when boundaries are unclear.
- Dev evidence is a CPU fp32 decode, not the GPU bf16 one; the measured difference is 0.1% of phonemes.
- One held-out replay on short clips. Long-session false warnings still need phase 2.

Files: `local-window-v1-dev.json` and `local-window-v1-dev-high.json` (train sweep), `local-window-v1-teacher.json` and `local-window-v1-int8.json` (held-out, per false positive and newly missed). Rerun: `tsx src/bench/local-window-cli.ts --split eval --nats off,10 [--evidence <dir>]` from `packages/core`.
