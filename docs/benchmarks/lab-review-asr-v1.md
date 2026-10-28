# Independent audio opinion: calibration on lab-v1.1 (ticket 12)

**Decision: fail. Listening stays fully manual.** Neither independent ASR model agrees with your 40 listens anywhere near the 95% bar. The best result was 8/40 under the fixed rule, and 20/40 even with the exploratory wider windows. Both models do flag wrong texts, but mostly because they reject nearly everything, correct audio included. As a pre-screen they would send you 38 of the 40 items anyway.

- Calibration set: queue `lab-v1.1/review/v1-check.json` (20 splices, 20 cases, with whole-word windows) and your verdicts in `/lab/reviews/lab-v1.1/me.json`. All 40 are `ok`. The one note ("the clip cuts off before حميم") was fixed by the whole-word windows; `حَمِيمٍ` is now in that item's text and audio.
- Per-item records are on the `tilawa-lab` volume in `/lab/reviews/lab-v1.1/`:
  - `asr-qwen3-asr-1.7b.json` and `asr-whisper-large-v3.json`: model id and revision, config, prompt (none), and per item the audio sha256, raw output, transcript, decision and word alignment for the labelled text and both controls.
  - `asr-calibration.json`: the summary.
  - The `*.pad1.json` files hold the exploratory run.
- Committed copies of the summary tables: `lab-review-asr-v1.json` and `lab-review-asr-v1-pad1.json` (this folder).
- Rerun: `modal run modal/review_asr.py::main` (add `--pad 1` for the exploratory run). Transcripts are cached by audio sha256, so a rerun needs no GPU unless you pass `--force`.

## What ran

| Model | Revision | Where | Config |
| --- | --- | --- | --- |
| `Qwen/Qwen3-ASR-1.7B` (qwen-asr 0.0.6) | `7278e1e70fe206f11671096ffdd38061171dd6e5` | Modal GPU from the L4 / A10 / L40S list (A10 for the decision run, L40S for the padded run), bf16 | language Arabic, no context or prompt, max 512 new tokens |
| `Systran/faster-whisper-large-v3` (faster-whisper 1.2.1, CTranslate2 4.8.4) | `edaa852ec7e145841d8ffdb056a99866b5f0a478` | Modal GPU, float16 | language ar, beam 5, no previous-text conditioning, no VAD, word timestamps |

Neither model is related to the teacher (`obadx/muaalem-model-v3_2`) or the student. Each transcriber receives only the item id and its audio window: 16 kHz mono PCM16 sliced from the session WAV, with the sha256 of exactly those bytes saved. It never receives the text, the label or a prompt. The comparison runs afterwards, in `tools/review_asr_core.py`:

- Both sides are normalised: diacritics and Quranic marks stripped, alef, hamza, ya and ta-marbuta forms unified, and the superscript alef tried both as ا and as nothing.
- The transcript is aligned to the labelled text word by word. One word may be heard as two, or two as one, but only on a near-exact match (similarity at least 0.9). One stray word is allowed at each edge.
- A word matches at similarity 0.75 or above and is a different word below 0.5. Anything in between is doubtful.
- **mismatch**: a word inside the window is wrong or missing, or an extra word of two or more letters is heard.
- **unsure**: only doubtful words, or labelled words missing at the very start or end of the window.
- **match**: every labelled word heard, and nothing extra inside.
- **combined**: match only if both models say match; mismatch if at least one says mismatch and neither says match; otherwise unsure.

The rule was committed (`033609f`) before any transcript existed. Its two fixes, stricter merges and a one-word edge allowance, came from running it against the true text as a perfect transcript (the **oracle** row), not from model output.

## Negative controls

All 40 of your verdicts are `ok`, so the 40 listens alone can't show that a model catches bad cases. Each audio window is therefore also paired with text that is wrong by construction. The transcript is the same one, because the models never see the text.

- **one-edit** (40): the labelled text with its planted edit undone, so the text claims clean where the audio has the edit. A planted wrong word goes back to the expected word, an added word or repeat is removed, and an omitted word or skipped ayah is put back. That covers the 20 splices and 6 repeats or restarts. The 14 clean windows instead get a one-word substitution, insertion or deletion planted in the text (seeded, using words clearly unlike the original).
- **one-word** (31): the subset of one-edit controls that differ by a single word. It leaves out skipped ayahs, repeats and restarts.
- **shifted** (40): the text of a different item, a gross mismatch.

Two terms in the tables below:

- **Flagged** means the model did not accept the wrong text: it said mismatch or unsure, and either way the item would go to you.
- **Agreement** means the model accepted your `ok` item as a match. Because all 40 verdicts are `ok`, agreement is the same as specificity.

Ranges are 95% Wilson intervals.

## Results (fixed rule, the decision)

| Who | Agreement (match on your 40 ok) | Called mismatch | Unsure | One-edit flagged | One-edit called mismatch | One-word flagged | Shifted flagged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-ASR-1.7B | 8/40 (20%, 10–35%) | 13 | 19 | 40/40 (100%, 91–100%) | 38 | 31/31 (100%, 89–100%) | 40/40 |
| Whisper large-v3 | 7/40 (18%, 9–32%) | 15 | 18 | 36/40 (90%, 77–96%) | 32 | 31/31 (100%, 89–100%) | 40/40 |
| Both (combined) | 6/40 (15%, 7–29%) | 21 | 13 | 40/40 (100%, 91–100%) | 34 | 31/31 | 40/40 |
| Oracle (true text as transcript) | 40/40 (100%, 91–100%) | 0 | 0 | 40/40 (100%, 91–100%) | 37 | 31/31 | 40/40 |

Read the flagged columns together with agreement. A model that rejects everything would also flag 100% of the negatives. The oracle row shows the comparison rule is not the bottleneck: with a perfect transcript it accepts all 40 real texts and flags all 40 wrong ones. Three of those 40 come out as unsure, not mismatch, because the planted word is a look-alike: بِحُسْبَانٍ for تُكَذِّبَانِ, ٱلْوَرِيدِ for ٱلْحَصِيدِ, تَنطِقُونَ for تَنَاصَرُونَ.

Why the models reject correct recitation, counted over your 40 `ok` items:

| | Qwen3 | Whisper |
| --- | --- | --- |
| Match | 8 | 7 |
| Only the first or last word of the window dropped | 10 | 12 |
| A misheard or doubtful word inside the window (often the phone copies, e.g. قراطيس heard as خارقة, رسوله heard as ارتو) | 22 | 11 |
| Non-recitation hallucinated: "اشتركوا في القناة", "ترجمة نانسي قنقر" | 0 | 7 |
| A planted repeat collapsed to the canonical text (the repeated words not transcribed) | 0 | 3 |

The last row is the dangerous failure. On the one-edit control, Whisper accepted the text with the repeat or restart removed on 4 of 6 items (`e08-x1`, `p05-x6`, `e05-x7`, `e12-x3`). It writes down the verse it expects, not what was said.

## Exploratory: 1 s wider audio (in-sample, not the decision)

The first table suggested the tight whole-word windows cost a lot, so the models were run again on audio widened by 1 s each side. The session's own words in the widened part are given to the comparison as context that never counts. The rule is otherwise unchanged. This was designed after seeing the first results, so it is in-sample.

| Who | Agreement | One-edit flagged | One-word flagged |
| --- | --- | --- | --- |
| Qwen3-ASR-1.7B | 17/40 (42%, 29–58%) | 38/40 (95%, 84–99%) | 30/31 |
| Whisper large-v3 | 20/40 (50%, 35–65%) | 36/40 (90%, 77–96%) | 31/31 |
| Both (combined) | 9/40 (22%, 12–38%) | 39/40 (98%, 87–100%) | 31/31 |
| Either model says match (most lenient) | 28/40 (70%) | 35/40 (88%) | 30/31 |

Padding fixes the dropped edge words: from 10 to 2 items for Qwen and from 12 to 2 for Whisper. Even the most lenient rule only reaches 70% agreement. It also still lets through all four repeat-removed texts, plus a one-word deletion (`p19-phone-5`) that Qwen missed because it merged أَو into the next word.

Padding also weakens the control for repeats that sit at a window edge. The extra copy can be absorbed into the context, so the oracle accepts 3 repeat-removed texts in this mode. If the window question is revisited, the fix belongs in the queue builder (1 s of lead-in and lead-out in the clip itself), not in the comparison.

## The bar, and what an opinion could ever cover

Proposed bar for any future model, on a fresh labelled set:

- agreement of at least 95% on the human `ok` items;
- one-edit flagged of at least 90%, with no misses on repeats or restarts;
- one-word flagged of at least 90%.

Report agreement and flagged together, never one alone. With 40 items, even 40/40 only gives a lower bound of 91%. A pass should therefore be confirmed by the 10% spot checks over the next one or two versions before it is trusted.

Even a passing ASR opinion only checks **words**: memorization errors, and splices that change, drop or add words. It cannot vouch for:

- **seams**: a click, a level or room-tone jump, a clipped breath or onset, an unnatural join or rhythm at the cut. Every splice still needs a listen, or an automated seam check that has been shown to catch these;
- **pronunciation and tajweed** (madd, ghunnah, qalqalah, makharij): a plain transcript says nothing about them;
- **audio damage** in the phone copies beyond intelligibility, e.g. distortion or a noise burst.

## Pre-screen mode (built, not to be used yet)

`modal run modal/review_asr.py::prescreen --queue <testSet>/review/<name>.json [--models qwen3-asr-1.7b,whisper-large-v3] [--spot 0.10] [--seed 12] [--seams listen|skip]`

It writes `/lab/data/<testSet>/review/<name>.prescreen.json` in the ticket 06 queue format. The queue holds every ASR mismatch and unsure item, a seeded 10% (rounded up) of the auto-cleared matches, and, with the default `--seams listen`, every splice. Each kept item's `context` ends with the reason it is there. Per-model records go to `/lab/reviews/<testSet>/asr-<model>.<name>.json`.

On `v1-check` today it keeps 38 of 40 items (21 mismatch, 13 unsure, 3 seams, 1 spot check), so it saves nothing. The test output was deleted from the volume so the review page doesn't list it. Use it only with a model that has passed the bar above.

## What could pass (speculative)

- **Verification, not open transcription.** For test-set splices the question is binary: did the audio say the labelled text A or the original B? An independent CTC model (a public Arabic or Quran wav2vec2 / XLS-R checkpoint, unrelated to the teacher) could force-align both candidates and compare their likelihoods. That avoids the ASR's spelling noise and its canonical-verse prior for free decoding. Scoring both candidates the same way is not the anchoring the audio-review rules warn about, but it does expose the candidates, so it should be calibrated with the same controls.
- **A Quran-fine-tuned Whisper** would transcribe recitation far better, but its prior pulls even harder toward the canonical verse. That is exactly what the repeat and one-edit controls caught here, so expect it to fail them unless it's checked carefully.
- It may not be worth it. A full listen of 40 items is about 15 minutes per test-set version. The pre-screen has a ceiling of about 10 minutes saved per version, since splices need a seam listen anyway. For training-data samples, which have no splices, the saving is larger.
