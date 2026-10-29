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

## Nemotron 3.5 Transcribe (2026-10-29)

**Decision: as an independent opinion it fails (23/40). Listening stays manual for anything that needs an independent check, which covers every clean window and all training-data samples.** A second arm, in which the model scores the labelled text against the original text instead of transcribing, passes the bar's numbers in-sample: 39/40 agreement, 39/40 one-edit and 30/31 one-word controls flagged, no repeat misses. But it is a likelihood comparison, not an opinion. It only applies to windows with a known alternative text, and its one miss was a planted word losing to the canonical one. It is wired as the default pre-screen mode. On `v1-check` you would still hear 35 of 40 windows.

### What ran

| Model | Revision | Where | Config |
| --- | --- | --- | --- |
| `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` (Transformers `144ecf0`, 5.22.0.dev0; torch 2.15) | `9d8cd6b79bfdd427b6558a5fdbf76fcaf8c75341` | Modal H100 80GB, bf16 (weights cached on the `tilawa-hf-cache` volume) | greedy, reasoning off, no system prompt, max 512 new tokens, the model card's loop guardrail |

- The model card's quick start serves the model with vLLM and NeMo Speech. Its in-process Transformers path uses the same weights, gives teacher-forced scores for the second arm, and speed doesn't matter for 40 short clips, so that is what ran. The Mamba layers ran on the PyTorch reference kernels (`mamba_ssm` was not installed). That is slower but numerically the reference.
- Independence: the model is unrelated to the teacher and the student. The checkpoint ships a Qwen3-ASR licence (`licenses/Qwen3-ASR-LICENSE.txt`), so part of it may derive from Qwen3-ASR. Treat its errors as possibly correlated with Qwen's, not with the teacher's.
- Arm 1 is label-blind, like the earlier models: the transcriber gets the item id and the same 16 kHz window bytes (same sha256), plus one fixed instruction. That instruction is the model card's verbatim ASR prompt with the language named: `Audio transcription. Language: Arabic. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.` The comparison rule and the controls are the ones above, unchanged.
- The code and both arms' rules were committed (`37ffc4d`) before any calibration window was transcribed. The only earlier output was a smoke test on a window outside the queue.

### Arm 1: label-blind transcription (fixed rule, the decision)

| Who | Agreement (match on your 40 ok) | Called mismatch | Unsure | One-edit flagged | One-edit called mismatch | One-word flagged | Shifted flagged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-ASR-1.7B (above) | 8/40 (20%, 10–35%) | 13 | 19 | 40/40 (100%, 91–100%) | 38 | 31/31 | 40/40 |
| Whisper large-v3 (above) | 7/40 (18%, 9–32%) | 15 | 18 | 36/40 (90%, 77–96%) | 32 | 31/31 | 40/40 |
| **Nemotron 3.5 Transcribe** | **23/40 (58%, 42–71%)** | 12 | 5 | **38/40 (95%, 84–99%)** | 35 | **29/31 (94%, 79–98%)** | 40/40 |
| Nemotron, recitation prompt (exploratory) | 26/40 (65%, 50–78%) | 11 | 3 | 40/40 (100%, 91–100%) | 37 | 31/31 | 40/40 |
| Nemotron CTC head, greedy (exploratory) | 0/40 | | | 40/40 | | 31/31 | 40/40 |

- Nemotron is the best free transcriber so far, about three times Qwen's agreement, and the first one that never collapsed a repeat: all 6 repeat and restart windows were written with the repeat. It is still 15 matches short of the 38/40 bar.
- Both one-edit misses are added-word splices where the model did not write the planted word. On `p15-x8` it wrote the canonical verse (`إنه لحق مثل ما أنكم`, with `قَوْمٌ` gone). On `e19-x12` it blended `إِنَّا كَيْفَ كُنَّا` into `إننا كفنّا`, which the rule reads as the original text.
- The exploratory prompt was the one suggested in the task: `Transcribe this Quran recitation verbatim, exactly as recited, including repetitions and mistakes. Language: Arabic. Output: transcript only.` It caught both added words and matched 3 more windows. It also moved the hallucinations around (it invented a continuation after `لحق` on `p15-x8` and a dua loop on `e15-phone-229`), so the two prompts differ mostly in where they go wrong. It was chosen in advance but is in-sample as a comparison.
- The checkpoint's CTC timestamp head can be read greedily from the audio encoder alone, with no text prior. It is not a recognizer: its output is fragments such as `يكبب بي المجون كعلام`, with stray non-Arabic characters.

Why Nemotron rejects correct recitation, counted over your 40 `ok` items:

| | Nemotron |
| --- | --- |
| Match | 23 |
| Only the first or last word of the window dropped (`p12-x7`; the leading restart fragment of `e05-x7`) | 2 |
| Planted word dropped or replaced by the canonical text (`p15-x8`, `p09-x8`, `e19-x12`) | 3 |
| A misheard word inside the window (e.g. `ٱلْوَرِيدِ` heard as `وعينة`, `كَٱلْأَعْلَـٰمِ` as `كلا علام` at the seam, `تَجْعَلُونَهُۥ` as `يدعونهم` and an inserted `لا` on phone copies) | 9 |
| Hallucinated text | 3 |
| A planted repeat collapsed to the canonical text | 0 |

- The hallucinations are Quran- and dua-flavoured, not YouTube boilerplate like Whisper's:
  - `e15-x8`: a dua (`اللهم اغفر لي ولوالدي وللمؤمنين ...`) that loops to the 512-token limit, too few repeats for the guardrail;
  - `e15-phone-229`: a pastiche of unrelated verses (`قل يا أيها الكافرون لا أسألكم عليه أجرا إن أجري إلا على الله ...`) for 55:77–78;
  - `p03-studio-5`: an invented continuation (`اتبع الحج والهدى`) past the window end.
- Phone copies: 3 of 7 rejected (2 mishearings, 1 hallucination), against 14 of 33 studio windows.

### Exploratory: 1 s wider audio (in-sample, not the decision)

| Who | Agreement | One-edit flagged | One-word flagged |
| --- | --- | --- | --- |
| Qwen3-ASR-1.7B (above) | 17/40 (42%, 29–58%) | 38/40 | 30/31 |
| Whisper large-v3 (above) | 20/40 (50%, 35–65%) | 36/40 | 31/31 |
| Nemotron 3.5 Transcribe | 22/40 (55%, 40–69%) | 36/40 (90%, 77–96%) | 30/31 |

For Nemotron, padding fixes the edge drops (`p12-x7` and `e08-x1` now match) but the extra context makes it write the expected verse more often:
- it collapsed the `p15-x5` repeat (`إنهم كانوا قبل ذلك محسنين`);
- it replaced the planted `بِحُسْبَانٍ` on `e20-x10` with the canonical `تكذبان`;
- it let the `p09-x8` revert through.

Two of its four one-edit misses (`e01-x4`, `e05-x7`) are the padded-oracle artefact described above. More context is not a free fix for this model.

### Arm 2: which text fits better (likelihood comparison, not an opinion)

The model never transcribes in this arm. Per window it scores given texts against the audio:

- **What was exposed:** the labelled text, its one-edit control text, its shifted control text, and its own arm-1 transcript.
  - All are undiacritised modern spellings (`plain_candidates` in `tools/review_asr_core.py`). A word that matches a transcript word takes the transcript's spelling; other words try their spelling variants, best one kept.
  - The model never saw the label type or the planted-edit fields as such. But for edited windows the one-edit text is the original text, so the model saw both candidates.
  - This ran only after arm 1 was saved. Because the candidates are shown, it weakens independence: it measures which given text the model finds likelier, not what it heard.
- **Scores:**
  - **LM:** the language model's log-likelihood of the text as its reply to the same transcription request (teacher-forced, through end of turn).
  - **CTC:** the CTC timestamp head's forced-alignment log-likelihood of the text's alignment-tokenizer pieces. This head reads the speech encoder only, so it has no text prior.
- **Rules**, fixed before the run, margin 2 nats:
  - **Pairwise:** the claimed text must beat the alternative by at least 2 nats (match); at least 2 nats worse is a mismatch; anything in between is unsure. The labelled text and its one-edit control are each other's alternative; the shifted text is compared with the labelled text.
  - **Near-best:** the claimed text must be within 2 nats of the model's own best hypothesis (its transcript for the LM, its greedy path for the CTC head).

| Score, rule | Agreement | One-edit flagged | One-word flagged | Shifted flagged | Repeat or restart misses |
| --- | --- | --- | --- | --- | --- |
| **LM, pairwise** | **39/40 (98%, 87–100%)** | **39/40 (98%, 87–100%)** | **30/31 (97%, 84–99%)** | 40/40 | none |
| LM, near-best | 30/40 (75%, 60–86%) | 33/40 (82%, 68–91%) | 25/31 (81%) | 40/40 | `p05-x6` |
| CTC, pairwise | 36/40 (90%, 77–96%) | 37/40 (92%, 80–97%) | 30/31 | 40/40 | `e08-x1`, `e12-x3` |
| CTC, near-best | 0/40 | 40/40 | 31/31 | 40/40 | none |

How to read the LM pairwise row:

- **Agreement and one-edit flagged are the same 40 comparisons read both ways.** A window agrees when the labelled text wins by 2 nats or more. Its control is flagged when the labelled text doesn't lose by 2 nats or more. That is 40 pairwise tests, of which 39 came out right, not 80 independent checks.
- **Only the 26 edited windows (20 splices, 6 repeats or restarts) have a real alternative text.** On those it is 25/26 (96%, 81–99%). The 14 clean windows were compared with a synthetic one-word change of their own text. In use, a clean window has no alternative, and the rule that needs none, near-best, fails (75%).
- **The miss is the dangerous kind.** `p12-x7` splices `مِهَـٰدًا` (78:6) in place of `كِفَاتًا` (77:25). The model prefers the canonical `كفاتا` by 3.3 nats, although its own transcript wrote `كهادا`. A mislabelled window like this would be cleared.
- **The margin is thin.** The median gap is 30 nats, but the smallest correct ones are 2.6 (`p09-x8`), 3.5 (`e08-x1`, a repeat), 4.7 and 4.9 (`p15-x5`, a repeat). At a 5-nat margin agreement would drop to 35/40, though controls would stay flagged.

The audio-only CTC head misses two repeats (`e12-x3` by 18.7 nats), so lacking a text prior did not make it safer on repeats. Near-best fails for both scores. For the CTC head, the greedy path is far likelier than any real text, so nothing matches.

### Pre-screen

`modal run modal/review_asr.py::prescreen --queue <testSet>/review/<name>.json` now defaults to `--models nemotron-likelihood`:

1. Nemotron transcribes every window label-blind; the transcripts are used only for spelling.
2. For each window with a planted edit, it scores the labelled text against the original text (the edit undone). A labelled text that wins by 2 nats is cleared.
3. Clean windows have no alternative text, so they stay unsure ("no alternative text") and go to the listener.
4. Splices keep their seam listen, and a seeded 10% of the cleared windows is spot-checked.

Records go to `/lab/reviews/<testSet>/asr-nemotron-likelihood.<name>.json`. The old two-model mode is still available with `--models qwen3-asr-1.7b,whisper-large-v3`.

On `v1-check` it keeps **35 of 40** windows:
- 19 splices for the seam listen;
- 14 clean windows;
- 1 mismatch (`p12-x7`);
- 1 spot check, from the 6 cleared repeat and restart windows.

That saves about 2 minutes per test-set version. The test output was deleted from the volume. Trust it only after the 10% spot checks hold over the next one or two versions; with 40 items the lower bound is 87%. It gives nothing for training-data samples, which have no alternative text.

### Cost and records

- An H100 80GB loads the checkpoint in about 71 s. The 40 windows took 89 s to transcribe (39 s of it the runaway dua loop) and 34 s to score.
- The weights were downloaded once on CPU (about 4 min, 67.5 GB).
- All GPU runs, including the smoke tests, the exploratory runs and the pre-screen check, came to about 20 min of H100 wall time, roughly $1.5 at about $4 an hour (estimate; check current Modal pricing).
- Records on the `tilawa-lab` volume in `/lab/reviews/lab-v1.1/`:
  - `asr-nemotron-3.5-transcribe.json`: per item the audio sha256, prompt, raw output with token ids, the CTC greedy path, transcript and decisions. Under `likelihood`, the exposed texts, every candidate spelling's LM and CTC score, and the arm 2 decisions.
  - `asr-nemotron-recitation-prompt.json` and `asr-nemotron-3.5-transcribe.pad1.json`: the exploratory runs.
  - `asr-calibration.nemotron-3.5-transcribe[.pad1].json`, `asr-calibration.nemotron-recitation-prompt.json` and `asr-calibration.nemotron-3.5-transcribe-likelihood.json`: the summaries.
- Committed tables in this folder: `lab-review-asr-v1-nemotron-3.5-transcribe[-pad1|-likelihood].json` and `lab-review-asr-v1-nemotron-recitation-prompt.json`.
- Rerun:
  - `modal run modal/review_asr.py::main --models nemotron-3.5-transcribe --name nemotron-3.5-transcribe`
  - then `modal run modal/review_asr.py::likelihood_main`
  - Arm 1 transcripts are cached by audio sha256, revision and prompt; the likelihood arm always runs on the GPU.

## NVIDIA Arabic FastConformer (2026-10-29)

**Decision: fails both arms and is not wired. The default pre-screen stays `nemotron-likelihood`.** Free transcription matches 23 of your 40 `ok` windows. The CTC likelihood arm reaches 36/40 with a repeat miss. As an independent data check it confirms every omission, skipped ayah and mutashabihat swap, and most wrong words. It does not hold up on added words and repeats: its Quran-trained prior deletes inserted speech, and 3 of its studio failures are windows you heard as correct.

### What ran

| Model | Revision | Where | Config |
| --- | --- | --- | --- |
| `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` (NeMo 2.7.3, torch 2.15) | `7f32349d952f42a28dce979ba73270aa2bbdfa89` | Modal T4, float32, TF32 off | hybrid Transducer + CTC, ~115M params, CC-BY-4.0; greedy for both decoders (RNNT `greedy_batch`, max 10 symbols; CTC greedy), batch 16 |

- Independence: the model is unrelated to the teacher and the student. Its training data includes `tarteel-ai/everyayah` (390 h of canonical recitation), so its prior is the Quran text. The held-out lab reciters exclude everyayah reciters, so the voices are unseen.
- Arm 1 is label-blind: the item id and the same 16 kHz window bytes as the other models (all 40 sha256 identical to Nemotron's), with no prompt. The comparison rule and the controls are unchanged. RNNT, the model's default decoder, is the decision decoder; CTC greedy and the two combined are reported too. The rule's normalisation already strips the model's harakat.
- The code and every rule below (arm 2 and the data check included) were committed in `5d019b0` before any calibration window was transcribed. The only earlier output was a smoke test on `e20` 0–8 s, a window outside the queue.
- NeMo 2.7.3's `transcribe()` leaves the encoder and the CTC head in train mode (dropout on) while `model.training` still reads False. Any log-probability computed after a `transcribe()` call is therefore noisy, and in the first smoke test it flipped the CTC path between diacritised and plain spellings. `modal/review_fastconformer.py` calls `eval()` after every decode and asserts eval mode before scoring. The smoke test caught this; no calibration output was affected.

### Arm 1: label-blind transcription (fixed rule)

| Who | Agreement (match on your 40 ok) | Called mismatch | Unsure | One-edit flagged | One-edit called mismatch | One-word flagged | Shifted flagged | Repeat misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-ASR-1.7B | 8/40 (20%, 10–35%) | 13 | 19 | 40/40 (100%, 91–100%) | 38 | 31/31 | 40/40 | none |
| Whisper large-v3 | 7/40 (18%, 9–32%) | 15 | 18 | 36/40 (90%, 77–96%) | 32 | 31/31 | 40/40 | 4 (`e08-x1`, `p05-x6`, `e05-x7`, `e12-x3`) |
| Nemotron 3.5 Transcribe | 23/40 (58%, 42–71%) | 12 | 5 | 38/40 (95%, 84–99%) | 35 | 29/31 (94%, 79–98%) | 40/40 | none |
| **FastConformer, RNNT** | **23/40 (58%, 42–71%)** | 3 | 14 | **39/40 (98%, 87–100%)** | 31 | **31/31 (100%, 89–100%)** | 40/40 | **`p05-x6`** |
| FastConformer, CTC greedy | 19/40 (48%, 33–62%) | 5 | 16 | 39/40 (98%, 87–100%) | 34 | 31/31 | 40/40 | `p05-x6` |
| FastConformer, both decoders | 18/40 (45%, 31–60%) | 5 | 17 | 39/40 | 34 | 31/31 | 40/40 | `p05-x6` |

- It ties Nemotron on agreement with a model about 260 times smaller, and it hallucinated nothing.
- The one control it let through is the dangerous kind. On `p05-x6` it wrote `فَثَبِّتُوا الَّذِينَ آمَنُوا` once, so the text with the repeat removed reads as a match.
- Most rejections are unsure, not mismatch (14 of 17). They are dropped edge words, which is why its agreement sits far below its control sensitivity.

Why FastConformer (RNNT) rejects correct recitation, counted over your 40 `ok` items:

| Failure mode | Items | Count |
| --- | --- | --- |
| Match | | 23 |
| Partial or dropped edge word (window-edge words missing or merged with a neighbour's sound) | `e13-x4`, `e14-x4`, `p10-x6`, `p13-x12`, `e05-x7` (leading restart fragment), `e12-x3`, `e02-studio-85`, `p03-studio-5`, `p19-phone-5`, `e15-phone-229` (the `فَبِ` of `فَبِأَىِّ` merged into `آلاء`) | 10 |
| Genuine mishearing, at a seam or on a phone copy | `p09-x8` (planted `قُوَّةً` at the seam heard as `أخوت`), `e10-studio-5` (`وإذا كالوهم` as `ألا يالوهم`), `e07-phone-293` (`قَرَاطِيسَ` as `قَضَاطِعَ`, `وَتُخْفُونَ` as `وَتُبْوُونَ`) | 3 |
| Planted word dropped, or output cut short | `e15-x8` (planted `لِلْأَنَامِ` dropped), `p15-x8` (`لحق مثل` only), `e19-x12` (`فَأَغْوَيْنَاكُمْ` only) | 3 |
| Canonicalisation (planted repeat collapsed to the verse) | `p05-x6` | 1 |
| Orthography or normalisation | | 0 |
| Hallucination | | 0 |

- The CTC head adds 5 rejections and recovers 1. Two of the five are CTC letter doubling (`مسسلممة`, `العصصفْفِ`), which is orthographic, and three are clipped words (`ون` for `عَنِ`, `وول` for `وَأُو۟لَـٰٓئِكَ`, `فيه` for `فِيهِمَا`).
- Phone copies: 3 of 7 rejected (1 mishearing, 2 edge), the same rate as Nemotron.

### Arm 2: which text fits better (CTC likelihood, not an opinion)

The model scores given texts with its CTC head: the forced-alignment log-likelihood of the text's SentencePiece pieces, summed over alignments. That head reads the audio encoder only. Unlike Nemotron's LM score, it has no autoregressive text model, but its encoder was trained on canonical recitation.

- **What was exposed:** per window, the labelled text, its one-edit control and its shifted control. Never the label type.
- **Spellings**, fixed in `5d019b0`:
  - **diac (the decision reading):** `diacritised_candidates` in `tools/review_asr_core.py`. Uthmani forms are turned into modern spelling that keeps the harakat; Quranic marks are dropped; the dagger alef is tried as `ا` and as nothing. A word shared with the window's own label-blind CTC greedy text or RNNT transcript takes that spelling. The best spelling is kept.
  - **plain:** the same with no harakat (`plain_candidates`).
  - The model writes with harakat, and its CTC head charges tens of nats for a spelling it wouldn't write (the smoke window's own RNNT text scored −118 against −0.7 for the greedy path). So the diacritised forms are the model's natural writing system, and both candidates of a pair get the same treatment.
- **Rules:** the same as Nemotron's. Pairwise means the claimed text must beat the alternative by at least 2 nats; at least 2 nats worse is a mismatch; anything in between is unsure. Near-best means within 2 nats of the CTC greedy path.

| Reading, rule | Agreement | One-edit flagged | One-word flagged | Shifted flagged | Repeat or restart misses |
| --- | --- | --- | --- | --- | --- |
| Nemotron LM, pairwise (above) | 39/40 (98%, 87–100%) | 39/40 (98%, 87–100%) | 30/31 (97%) | 40/40 | none |
| **FastConformer CTC, diac, pairwise** | **36/40 (90%, 77–96%)** | **37/40 (92%, 80–97%)** | **29/31 (94%, 79–98%)** | 40/40 | **`p05-x6`** |
| FastConformer CTC, plain, pairwise | 35/40 (88%, 74–95%) | 35/40 (88%, 74–95%) | 28/31 (90%) | 40/40 | `p05-x6`, `e05-x7` |
| FastConformer CTC, diac, near-best | 10/40 (25%, 14–40%) | 40/40 | 31/31 | 40/40 | none |
| FastConformer CTC, plain, near-best | 7/40 (18%, 9–32%) | 40/40 | 31/31 | 40/40 | none |
| Both must clear: Nemotron LM and FastConformer diac, pairwise (exploratory, in-sample) | 35/40 (88%, 74–95%) | 40/40 (100%, 91–100%) | 31/31 | 40/40 | none |

- **It gets Nemotron's dangerous miss right.** On `p12-x7` the planted `مِهَادًا` beats the canonical `كِفَاتًا` by 30.0 nats; its greedy path wrote `كِهَادًا`.
- **It fails where its prior pulls hardest: inserted speech.** All 14 clean windows agree. The 4 edited windows that fail are:
  - `p05-x6`, a repeat: the text without the repeat wins by 98.9 nats. The greedy path has the phrase once; the head spends blanks on the repeated speech rather than emit it twice.
  - `p15-x8` (`قَوْمٌ`) and `e19-x12` (`كَيْفَ`), added words: the canonical text wins by 41.2 and 2.8 nats.
  - `p09-x8`, a wrong word at a seam: unsure, at 0.2 nats.
- On the 26 edited windows it is 22/26. The median winning margin is 50 nats, against 30 for Nemotron.
- Plain spelling is worse and adds a restart miss (`e05-x7`), which supports scoring with harakat.
- The two models miss different things: Nemotron misses the planted wrong word, FastConformer the repeat and the added words. Requiring both to clear flags every control, but agreement drops to 35/40, below the 38/40 bar. In pre-screen use it would clear 21 of 26 edited windows, against Nemotron's 25, so you would hear 4 more per version, but the `p12-x7` hole would be closed. This was designed after seeing both arms, so it's in-sample.

### Independent data check of every planted edit

Ticket 04's checks use the teacher, so they aren't independent. This check scores, for every planted edit in lab-v1.1 (320 mistakes plus 40 repeats and restarts, each in the studio and phone copies), the said text against the original text with FastConformer's CTC head.

- **Window:** the same as the teacher check: word before, edit, word after, using labelled times from the session JSON, padded 0.1 s (`edit_texts`).
- **Pass:** the said text wins by at least 3.0 nats (the teacher check's `LLR_MIN`), using diacritised candidates spelled against the window's own label-blind transcripts. An original with no CTC path, such as a skipped ayah too long for the window, counts as infinitely worse.
- The rule was committed in `5d019b0` before the run.

| Class | n | Teacher, studio (phone) | **FastConformer, studio** (phone) | FastConformer, 2 s context, studio (phone), exploratory |
| --- | --- | --- | --- | --- |
| wrong word | 94 | 94 (88) | **81** (77) | 83 (81) |
| mutashabihat | 10 | 10 (7) | **10** (10) | 10 (10) |
| omitted word | 42 | 42 (40) | **42** (42) | 42 (42) |
| added word | 105 | 105 (93) | **68** (54) | 77 (65) |
| skipped passage | 69 | 69 (65) | **69** (69) | 69 (69) |
| repeat | 23 | 23 (20) | **14** (11) | 18 (19) |
| restart | 17 | 17 (17) | **17** (14) | 11 (13) |
| **mistakes** | 320 | 320 (293) | **270 (84%)** (252) | 281 (267) |
| **repeats and restarts** | 40 | 40 (37) | **31** (25) | 29 (32) |

- **Studio against the teacher:**
  - 301 edits pass both checks and 59 pass only the teacher's. None passes only FastConformer's, and none fails both.
  - Every failure is a window the teacher hears cleanly, by 9–165 nats (median 47).
- **Phone against the teacher:**
  - 261 pass both, 69 only the teacher's, 16 only FastConformer's, and 14 fail both.
- **The failures look like the model, not the labels.**
  - 46 of the 59 studio failures are added words (37) or repeats (9), the classes where its prior deletes speech. The other 13 are wrong words.
  - 3 of them are windows you listened to and marked `ok`: `e19-x12` (added `كَيْفَ`), `e08-x1` and `p15-x5` (repeats).
  - Window length doesn't explain it: pass rates are flat from 2 s to over 4 s.
  - Empty transcripts don't explain it either: 48 windows got no RNNT text, and 35 of those still pass.
- **Secondary readings** (studio mistakes):
  - plain spelling 291/320, but repeats are worse (12/23);
  - llr > 0: 283/320;
  - the label-blind RNNT transcript is closer to the said text than to the original: 268/320.
  - 36 studio edits are supported by neither the likelihood nor the transcript: 21 added words, 8 wrong words and 7 repeats.
- **The 2 s context run** (exploratory, in-sample): shared whole words on each side of both texts. It helps added words and repeats a little (studio mistakes 281/320) but makes restarts worse (17 to 11): the extra copy at the edge gets absorbed, as in the padded runs above.

**What this says about the test set:** a second, unrelated listener confirms every omission, skipped ayah and mutashabihat swap in both copies, and 82–86% of wrong words. Its failures sit in the classes its prior can't hear, include windows a human confirmed, and never point at an edit the teacher was unsure of. Nothing here suggests a label error. It is not strong enough to vouch for added words and repeats on its own.

**Proposal for ticket 04** (not wired; the builder loop is unchanged): add `checks-fastconformer-v1.json` as an independent second opinion, reported beside the teacher's pass.
- Treat it as a hard second check only for omitted words, skipped passages, mutashabihat swaps and wrong words. In studio that is 202 of 215 passing, plus 13 wrong words for a listen.
- For added words, repeats and restarts, report it but don't gate on it. Route the edits that neither the likelihood nor the transcript supports to the listening queue instead: 28 added words and repeats in studio for lab-v1.1, about 7 minutes of listening.

### Cost and records

- T4, about 23 s to load. GPU compute was 4 s for arm 1, 8 s for arm 2, and 54 s for each of the two data-check runs (720 windows each).
- All GPU containers, including the image build, two smoke tests, three debug runs and the exploratory run, came to roughly 12 minutes of T4 time, about $0.15 (estimate; check current Modal pricing).
- Records on the `tilawa-lab` volume:
  - `/lab/reviews/lab-v1.1/asr-fastconformer-rnnt.json` and `asr-fastconformer-ctc.json`: arm 1 per item, with the audio sha256, both decoders' text, the CTC greedy path and every comparison;
  - `asr-fastconformer-likelihood.json`: arm 2 per item, with every spelling's score;
  - `asr-calibration.fastconformer.json` and `asr-calibration.fastconformer-likelihood.json`: the summaries;
  - `/lab/data/lab-v1.1/checks-fastconformer-v1.json`: the data check per edit and variant, with the teacher's result alongside, and `checks-fastconformer-v1.ctx2.json`: the exploratory run.
- Committed tables in this folder: `lab-review-asr-v1-fastconformer.json`, `lab-review-asr-v1-fastconformer-likelihood.json`, `lab-checks-fastconformer-v1.json` and `-ctx2.json`.
- Rerun:
  - `modal run modal/review_fastconformer.py::main` (arm 1; reuses saved transcripts unless `--force`);
  - `modal run modal/review_fastconformer.py::likelihood_main`;
  - `modal run modal/review_fastconformer.py::check_main [--context 2]`.


## Round 2 (2026-10-29): where Nemotron goes wrong, a prompt ablation, ROVER and rasm matching

All of this is on the same 40 `ok` windows and the same controls as above, so it is all in-sample. Part A uses only the saved transcripts. Part B adds one new prompt. Part C adds a looser comparison and a voting scheme. What was fixed before its own run, and what was tuned after seeing results, is stated per part.

### A. Error analysis: why Nemotron rejects correct recitation

Each window that Nemotron did not accept (17 with the decision prompt `verbatim`, 14 with the exploratory `recitation` prompt) was classified by hand from its saved transcript and word alignment into one root cause:

1. **Orthography only.** The model heard the word right and our comparison is too strict: Uthmani vs standard spelling, hamza forms, alif wasla, ta marbuta, alif maqsura, dagger alif, tanwin, or a word written joined or split.
2. **Edge word.** A word at the window start or end is dropped, or a neighbouring word bleeds in. By the session word times, every labelled word in this queue lies fully inside its window, with 0.04–0.25 s to spare. The neighbouring words end or start exactly at the window edge, 0 s away. So the only real "partial" words are the neighbours, whose coarticulated sound can leak in.
3. **Genuine mishearing.** The table notes which letters went wrong, whether it was at a splice seam, and whether it was a phone copy.
4. **Canonicalisation.** The planted edit was dropped or blended so the transcript reads as the original verse, or a restart was collapsed.
5. **Hallucination or loop.**

The diffs are on normalised words (diacritics stripped): `[label→heard]` is a substitution, `[-w]` a labelled word not heard, `[+w]` a heard word not in the text, `(w)` an edge word the rule allows. The machine-readable classification is in `lab-review-asr-v1-errors.json`.

#### Nemotron, decision prompt `verbatim`: the 17 windows it did not accept

| Item | Kind | Labelled (normalised) | Transcript | Diff | Bucket | Where / why |
| --- | --- | --- | --- | --- | --- | --- |
| `e20-x6` | splice wrong_word | يكذب بها المجرمون كالاعلام بينها وبين حميم | يُكَذِّبُ بِهَا الْمُجْرِمُونَ. كَلَّا عِلَامٌ بَيْنَهَا وَبَيْنَ حَمِيمٍ. | يكذب بها المجرمون [+كلا] [كالاعلام→علام] بينها وبين حميم | 1 | planted word, at the seam; split كَٱلْ + أعلام and alif wasla dropped: skeleton identical |
| `e20-x10` | splice wrong_word | الا ربكما بحسبان فيهن قاصرات | وَالْآبَاءُ رَبِّكُمَا بِحُسْبَانِ فِيهِنَّ قَاصِرَاتُ. | [الا→والابا] ربكما بحسبان فيهن قاصرات | 3 | first word ءالاء heard as والآباء, right after the cut (window starts where فبأي ends); seam word بحسبان heard right |
| `p09-x8` | splice wrong_word | بالجنه التي قوه توعدون | في الجنة التي توعدون. | (في) بالجنه التي [-قوه] توعدون | 4 | planted قوة dropped at the seam (and بـ heard as في) |
| `e13-x4` | splice wrong_word | جنات وحب الوريد والنخل باسقات | جَنَّاتٍ وَحَبًّا وَعِينَةٍ وَالنَّخْلَ بَاسِقَاتٍ. | جنات وحب [الوريد→وعينه] والنخل باسقات | 3 | planted الوريد heard as وعينة, at the seam |
| `e15-x8` | splice wrong_word | الا ربكما للانام مدهامتان | اللَّهُمَّ اغْفِرْ لِي وَلِوَالِدَيَّ وَلِمُؤْمِنِينَ الْقِرَاءَةَ وَالْمُؤْمِنِ … | [+اللهم] [+اغفر] [+لي] [+ولوالدي] (ولمومنين) [الا→القراه] [+والمومنين] [+والمومنات] [+والمسلمين] [ربكما→والمسل … | 5 | dua loop to the 512-token limit |
| `p15-x8` | splice added_word | انه لحق مثل قوم ما انكم | إِنَّهُ لَحَقٌّ مِثْلَ مَا أَنَّكُمْ. | انه لحق مثل [-قوم] ما انكم | 4 | planted قوم dropped: exactly the written 51:23 |
| `e19-x12` | splice added_word | فاغويناكم انا كيف كنا غاوين فانهم | فأغويناكم إننا كفنّا غاوين فإنهم. | فاغويناكم انا [-كيف] كنا غاوين فانهم | 4 | planted كيف blended: إنا كيف كنا → إننا كفنا, which reads as the original |
| `e10-x9` | splice skipped_passage | نضره النعيم ختامه مسك وفي | نضرة النعيم, كتاب مسك, وفي | نضره النعيم [ختامه→كتاب] مسك وفي | 3 | ختامه heard as كتاب, first word after the skip seam |
| `p12-x7` | splice wrong_word | الم نجعل الارض مهادا احيا | ألم نجعل الأرض كهادا؟ | الم نجعل الارض مهادا [-احيا] | 2 | last word أحياء dropped (inside the window, 0.25 s before its end); padding fixed it. 78:6 ends at مهادا, so maybe prior too (speculative) |
| `e08-x1` | repeat | ذالك لايه وما ذالك لايه وما كان اكثرهم مومنين | ذلك لآيتهم، وما ذلك لآيتهم، وما كان أكثرهم مؤمنين. | ذالك [لايه→لايتهم] وما ذالك [لايه→لايتهم] وما كان اكثرهم مومنين | 3 | لآيةً heard as لآيتهم on both copies (tanwin ending); not at a seam |
| `e01-x4` | restart | الارض بغير الحق واستكبر هو وجنوده في الارض بغير الحق وظنوا انهم | في الأرض بغير الحقّ واستكبروا وجنودُه في الأرض بغير الحقّ وظنّوا أنّهم. | (في) الارض بغير الحق واستكبر [-هو] وجنوده في الارض بغير الحق وظنوا انهم | 3 | واستكبر هو heard as one word واستكبروا (lost /h/); not at a seam |
| `p05-x6` | repeat | فثبتوا الذين امنوا فثبتوا الذين امنوا سالقي في قلوب | فَذَبِّطُوا الَّذِينَ آمَنُوا. فَذَبِّطُوا الَّذِينَ آمَنُوا سَأُلْقِي فِي قُلُوبِهِمْ. | [فثبتوا→فذبطوا] الذين امنوا [فثبتوا→فذبطوا] الذين امنوا سالقي في [قلوب→قلوبهم] | 3 | فثبتوا heard as فذبطوا on both copies (ث→ذ, ت→ط); قلوب → قلوبهم at the end |
| `e05-x7` | restart | اجري الا علي وما اسلكم عليه من اجر ان اجري الا علي رب العالمين | وَمَا أَسْأَلُكُمْ عَلَيْهِ مِنْ أَجْرٍ. إِنْ أَجْرِيَ إِلَّا عَلَى رَبِّ الْعَالَمِينَ. | [-اجري] [-الا] [-علي] وما اسلكم عليه من اجر ان اجري الا علي رب العالمين | 4 | restart's leading fragment أجري إلا على dropped: transcript is the written 26:109 |
| `p03-studio-5` | clean studio | انذرتهم ام لم تنذرهم لا يومنون انما تنذر من | أَأَنْذَرْتَهُمْ أَمْ لَمْ تُنْذِرْهُمْ لَا يُؤْمِنُونَ. إِنَّمَا تُنْذِرُ مَنِ  … | انذرتهم ام لم تنذرهم لا يومنون انما تنذر من (اتبع) [+الحج] [+والهدي] | 2 | abutting next word اتبع bled in (linked kasra of منِ), then invented الحج والهدى |
| `e07-phone-293` | clean phone | تجعلونه قراطيس تبدونها وتخفون كثيرا وعلمتم ما لم تعلموا | يدعونهم قراطيس تبذلونها وتخون كثيرا وعلمتم ما لم تعلموا. | [تجعلونه→يدعونهم] قراطيس تبدونها وتخفون كثيرا وعلمتم ما لم تعلموا | 3 | phone: تجعلونه → يدعونهم (also تبذلونها, وتخون, which the rule tolerated) |
| `p06-phone-173` | clean phone | حتي يغيروا ما بانفسهم واذا اراد | حتى لا يغيروا ما بأنفسهم، وإذا أرادوا. | حتي [+لا] يغيروا ما بانفسهم واذا [اراد→ارادوا] | 3 | phone: inserted لا (حتى لا, a common phrase) and أراد → أرادوا |
| `e15-phone-229` | clean phone | الا ربكما تكذبان تبارك اسم ربك | قُلْ يَا أَيُّهَا الْكَافِرُونَ لَا أَسْأَلُكُمْ عَلَيْهِ أَجْرًا ۖ إِنْ أَجْرِي … | [+قل] [+يا] [+ايها] (الكافرون) [الا→لا] [+اسالكم] [+عليه] [ربكما→اجرا] [تكذبان→ان] [تبارك→اجري] [اسم→الا] [ربك … | 5 | pastiche of other verses (قل يا أيها الكافرون ... وأنا من المنذرين) for 55:77–78 |

#### Nemotron, `recitation` prompt (exploratory): the 14 windows it did not accept

| Item | Kind | Labelled (normalised) | Transcript | Diff | Bucket | Where / why |
| --- | --- | --- | --- | --- | --- | --- |
| `e20-x6` | splice wrong_word | يكذب بها المجرمون كالاعلام بينها وبين حميم | يُكَذِّبُ بِهَا الْمُجْرِمُونَ. كَلَّا عِلَامٌ بَيْنَهَا وَبَيْنَ حَمِيمٍ. | يكذب بها المجرمون [+كلا] [كالاعلام→علام] بينها وبين حميم | 1 | same as the decision run |
| `e20-x10` | splice wrong_word | الا ربكما بحسبان فيهن قاصرات | إِلَىٰ رَبِّكُمَا فِي حُسْبَانِ فِي نَقْصِرَاتٍ. | الا ربكما [+في] بحسبان [فيهن→في] [قاصرات→نقصرات] | 3 | ءالاء → إلى, بحسبان split as في حسبان, فيهن قاصرات → في نقصرات (the ن moved), around the seam |
| `p09-x8` | splice wrong_word | بالجنه التي قوه توعدون | وَبِالْجَنَّةِ الَّتِي تُوعَدُونَ. | بالجنه التي [-قوه] توعدون | 4 | planted قوة dropped at the seam |
| `e15-x8` | splice wrong_word | الا ربكما للانام مدهامتان | إِنَّا آلَاءِ رَبِّكُمَا لِلْأَنَامِ مُدْهَامَّةً. | (انا) الا ربكما للانام [مدهامتان→مدهامه] | 3 | last word مدهامتان → مدهامة (dual ending); extra leading إنا was allowed |
| `p06-x1` | splice omitted_word | اولايك الذين بربهم واولايك | أُولَئِكَ الَّذِينَ ضَرَبْنَا أَيْدِيَهُمْ وَأَرْجُلَهُمْ وَأُولَئِكَ الَّذِينَ | اولايك الذين [+ضربنا] [+ايديهم] [بربهم→وارجلهم] واولايك (الذين) | 5 | invented ضربنا أيديهم وأرجلهم in place of بربهم, at the omitted-word seam |
| `p15-x8` | splice added_word | انه لحق مثل قوم ما انكم | إِنَّهُ لَحَقٌّ مِنْ رَبِّكُمْ أَنْ تَسْتَمِعُوا لِلْحَقِّ مِنْ رَبِّكُمْ فَأَنْ … | انه لحق [مثل→من] [قوم→ربكم] [+ان] [+تستمعوا] [+للحق] [ما→من] [+ربكم] [انكم→فانتم] (مستسلمون) | 5 | invented continuation after لحق at the seam: من ربكم أن تستمعوا للحق ... |
| `p10-x6` | splice added_word | تكذبان كل من عليها تكذبان فان ويبقي | تُكَذِّبَانِ. كُلُّ مَنْ عَلَيْهَا تُكَذِّبَانِ. فَالْوَاقِعُ. | تكذبان كل من عليها تكذبان [فان→فالواقع] [-ويبقي] | 3 | فانٍ ويبقى → فالواقع, last two words, after the added تكذبان |
| `e10-x9` | splice skipped_passage | نضره النعيم ختامه مسك وفي | نَبْضَةَ النَّعِيمِ هِتَامٌ مِسْكٌ وَفِيهِ. | [نضره→نبضه] النعيم [ختامه→هتام] مسك وفي | 3 | نضرة → نبضة, ختامه → هتام (first word after the skip seam) |
| `e01-x4` | restart | الارض بغير الحق واستكبر هو وجنوده في الارض بغير الحق وظنوا انهم | في الأرض بغير الحق، واستكبروا وجنودُه في الأرض بغير الحق، وظنوا أنهم | (في) الارض بغير الحق واستكبر [-هو] وجنوده في الارض بغير الحق وظنوا انهم | 3 | same as the decision run (هو lost) |
| `p05-x6` | repeat | فثبتوا الذين امنوا فثبتوا الذين امنوا سالقي في قلوب | فَذَبِّتُ الَّذِينَ آمَنُوا فَذَبِّتُ الَّذِينَ آمَنُوا سَأُلْقِي فِي قُلُوبِهِمْ | [فثبتوا→فذبت] الذين امنوا [فثبتوا→فذبت] الذين امنوا سالقي في [قلوب→قلوبهم] | 3 | فثبتوا → فذبت on both copies |
| `p03-studio-5` | clean studio | انذرتهم ام لم تنذرهم لا يومنون انما تنذر من | أَأَنْذَرْتَهُمْ أَمْ لَمْ تُنْذِرْهُمْ لَا يُؤْمِنُونَ إِنَّمَا تُنْذِرُ مَنِ ا … | انذرتهم ام لم تنذرهم لا يومنون انما تنذر من (اتبع) [+الحجور] | 2 | same bleed of اتبع, then invented الحجور |
| `e07-phone-293` | clean phone | تجعلونه قراطيس تبدونها وتخفون كثيرا وعلمتم ما لم تعلموا | يدعونهم فراقيس تبذلونها وتخون كثيرا وعلمتم ما لم تعلموا. | [تجعلونه→يدعونهم] [قراطيس→فراقيس] تبدونها وتخفون كثيرا وعلمتم ما لم تعلموا | 3 | phone: تجعلونه → يدعونهم, قراطيس → فراقيس |
| `p06-phone-173` | clean phone | حتي يغيروا ما بانفسهم واذا اراد | حتى لا يغيروا ما بأنفسهم، وإذا أرادوا. | حتي [+لا] يغيروا ما بانفسهم واذا [اراد→ارادوا] | 3 | phone: same inserted لا, أراد → أرادوا |
| `e15-phone-229` | clean phone | الا ربكما تكذبان تبارك اسم ربك | اللَّهُمَّ اغْفِرْ لِي وَلِوَالِدَيَّ وَلِمَنْ أَحْبَبْتَ، وَلِمَنْ أَحْبَبْتَ،  … | [+اللهم] (اغفر) [الا→لي] [+ولوالدي] [ربكما→ولمن] [+احببت] [تكذبان→ولمن] [تبارك→احببت] [اسم→ولمن] [ربك→احببت] ( … | 5 | dua loop: اللهم اغفر لي ولوالدي ولمن أحببت ×4 |

#### Summary per model

Qwen3 and Whisper were bucketed automatically, which is coarse:
- **5:** boilerplate text, or more than 3 extra words;
- **4:** an edited window whose one-edit control text matches the transcript;
- **1:** the space-free skeleton is equal;
- **2:** only edge reasons;
- **3:** anything else.

Applied to Nemotron, the same automatic pass agrees with the hand buckets except for two calls: `p09-x8` (planted word dropped) and `p03-studio-5` (neighbour bleed plus an invented continuation).

| Root cause | Nemotron `verbatim` (17) | Nemotron `recitation` (14) | Qwen3-ASR (32, automatic) | Whisper (33, automatic) |
| --- | --- | --- | --- | --- |
| 1 orthography only | 1 (`e20-x6`) | 1 (`e20-x6`) | 0 | 0 |
| 2 edge word | 2 (`p12-x7`, `p03-studio-5`) | 1 (`p03-studio-5`) | 10 | 11 |
| 3 mishearing | 8 | 8 | 22 | 11 |
| 4 canonical or planted word dropped | 4 (`p09-x8`, `p15-x8`, `e19-x12`, `e05-x7`) | 1 (`p09-x8`) | 0 | 4 (all repeats or restarts collapsed) |
| 5 hallucination or loop | 2 (`e15-x8`, `e15-phone-229`) | 3 (`p06-x1`, `p15-x8`, `e15-phone-229`) | 0 | 7 ("اشتركوا في القناة", "ترجمة نانسي قنقر") |
| **Would flip if only the comparison changed (buckets 1–2)** | **3 → at most 26/40** | **2 → at most 28/40** | 10 → at most 18/40 | 11 → at most 18/40 |
| Of those, flippable without ignoring audio that is inside the window | 2 (`e20-x6`, `p03-studio-5`) | 2 | 0 | 0 |

What this says:

- **The comparison is not the bottleneck.** Even crediting every bucket 1–2 case, Nemotron reaches at most 26/40 (`verbatim`) or 28/40 (`recitation`), against a bar of 38. Only one window, `e20-x6`, is a pure spelling problem: `كَٱلْأَعْلَـٰمِ` written `كلا علام` has the same skeleton.
- **Qwen's and Whisper's edge drops are not partial words.** The dropped words are fully inside the window, so ignoring them would ignore real audio. A comparison fix cannot honestly recover them. Wider windows can (see the 1 s run above).
- **About half of Nemotron's real errors sit at the seam.** In buckets 3–4 of the decision run, 5 of 12 are on or next to the planted word: `e13-x4`, `e10-x9`, `p09-x8`, `p15-x8`, `e19-x12`. These are exactly the words a reviewer cares about, and the model's language prior pulls them toward the verse it expects.
- **The rest are ordinary consonant confusions**, often the same error on both copies of a repeat: ث→ذ and ت→ط (`فثبتوا` → `فذبطوا`), a tanwin ending (`لآيةً` → `لآيتهم`), a lost /h/ (`واستكبر هو` → `واستكبروا`). Two phone copies also have word-level confusions.

Vivid examples:

- **`p15-x8`** (added word, decision run): the audio says `إِنَّهُۥ لَحَقٌّ مِّثْلَ قَوْمٌ مَآ أَنَّكُمْ` and the transcript is exactly 51:23, `إِنَّهُ لَحَقٌّ مِثْلَ مَا أَنَّكُمْ`. With the `recitation` prompt the planted word is gone again, this time replaced by an invented sermon: `مِنْ رَبِّكُمْ أَنْ تَسْتَمِعُوا لِلْحَقِّ مِنْ رَبِّكُمْ فَأَنْتُمْ مُسْتَسْلِمُونَ`.
- **`e15-phone-229`** (clean phone copy of 55:77–78): `verbatim` writes `قُلْ يَا أَيُّهَا الْكَافِرُونَ لَا أَسْأَلُكُمْ عَلَيْهِ أَجْرًا إِنْ أَجْرِيَ إِلَّا عَلَىٰ اللَّهِ وَأَنَا مِنَ الْمُنْذِرِينَ`, a pastiche of at least three other surahs. `recitation` writes a dua loop instead (`اللهم اغفر لي ولوالدي ولمن أحببت` ×4).
- **`e05-x7`** (restart): the recitation is `…أَجْرِىَ إِلَّا عَلَىٰ | وَمَآ أَسْـَٔلُكُمْ … رَبِّ ٱلْعَـٰلَمِينَ` and the transcript is the clean 26:109. The first, broken attempt vanished. That is the canonical prior again, this time on a window edge.
- **`e20-x6`** (the one comparison bug): `كَٱلْأَعْلَـٰمِ` was heard as `كَلَّا عِلَامٌ`, which is the right sound split into two words with the alif wasla dropped. The word-level rule called it an extra `كلا` plus a doubtful `علام`.
- **`p05-x6`** (repeat): `فَثَبِّتُوا۟` became `فَذَبِّطُوا` on both copies. That is not noise: the model consistently hears this reciter's ث as ذ and ت as ط.

### Preregistration for B and C (committed before either ran)

Everything below was written down and committed after Part A, and before any new transcript or any evaluation of a new rule existed. The 40 windows are the same ones Part A was read from, so all of B and C remains in-sample. "Preregistered" here means only that the rule and the run were fixed before their own numbers were seen.

**B. Prompt ablation.** Three new Nemotron prompts (`NEMOTRON_PROMPTS` in `modal/review_asr.py`). Together with `recitation` they make a 2 × 2 of the word "Quran" against the clause "including repetitions and mistakes":

| | with "including repetitions and mistakes" | without |
| --- | --- | --- |
| with "Quran" | `recitation`: Transcribe this Quran recitation verbatim, exactly as recited, including repetitions and mistakes. Language: Arabic. Output: transcript only. | `quran`: Transcribe this Quran recitation verbatim, exactly as recited. Language: Arabic. Output: transcript only. |
| without | `mistakes` (the one asked for): Transcribe this recitation verbatim, exactly as recited, including repetitions and mistakes. Language: Arabic. Output: transcript only. | `plain`: Transcribe this recitation verbatim, exactly as recited. Language: Arabic. Output: transcript only. |

- Same windows, model, revision, greedy decoding and round-1 comparison rule. Records go to `asr-nemotron-{mistakes,quran,plain}-prompt.json`.
- On 40 items, a difference of 3 or fewer matches is within noise, so a change counts only if the discordant pairs point one way.

**C1. The rasm rule** (`judge_rasm` in `tools/review_asr_core.py`):
- Both sides go to the consonantal skeleton (`rasm`):
  - all diacritics, Quranic marks and tatweel dropped;
  - every long-vowel alif dropped (ا, alif wasla, dagger alif; the Uthmani وٰ);
  - every hamza form unified to ء, kept as a consonant;
  - ى → ي, ة → ه.
- Word alignment is the round-1 one with two changes:
  - up to 3 labelled words may match up to 3 heard words as one space-free string, but only when the skeletons are identical (a looser merge would hide a dropped لا or أو);
  - thresholds are unchanged (0.75 / 0.5, one free stray word per edge).
- Edge words come from the session word times:
  - a session word ending or starting within 0.10 s outside the window becomes context that never counts;
  - so does a labelled edge word that sticks out of the window (none do in this queue).
  - Unlike the padded round-1 rule, heard words beyond that context and the one-word allowance still count as extra. This keeps a restart or repeat at the window edge from being absorbed.
- Reported with three variants:
  - `rasm-no-edge` (the edge part off);
  - `rasm-no-hamza` (hamza dropped too, the literal skeleton, as a sensitivity check);
  - `rasm-phonetic` (emphatic/plain, interdental and q/k pairs merged, ع dropped). This one is **exploratory**: it was chosen after seeing `فثبتوا` → `فذبطوا` in Part A.
- The rule was designed after reading Part A, so even the primary `rasm` row is tuned in-sample.

**C2. ROVER** (`rover` in `tools/review_asr_core.py`, `round2` in `modal/review_asr.py`):
- **Prompt:** `pick_prompt` chooses among `verbatim`, `recitation` and `mistakes` on the round-1 rule. It takes the most agreement among prompts whose controls hold (one-edit and one-word flagged ≥ 90%, no repeat misses); ties go to more one-edit flagged, then the earlier prompt in that order.
- **Perturbations:** each window is transcribed six times, at speed 0.9×, 1.0× and 1.1× (ffmpeg `atempo`, pitch kept) × {raw, denoised by DeepFilterNet3 at 48 kHz}. The unperturbed run is first.
- **Voting:**
  - The pivot is the medoid transcript (least total alignment cost to the others).
  - Every transcript is aligned to the pivot on the skeleton. It votes per pivot word (a word, a merged phrase, "covered" or nothing) and per gap (an inserted phrase or nothing).
  - Votes are clustered at skeleton similarity ≥ 0.75. A pivot word is written if its biggest cluster has at least as many votes as nothing and covered together. An insertion is written only if it outvotes the voters that inserted nothing.
  - Each written word takes its most common spelling.
- **Variant:** `rover+qwen+whisper` adds the saved Qwen3 and Whisper transcripts as two more voters.

**Ablation and the bar.**
- Rows: C1 alone (saved transcripts under `rasm`), C2 alone (ROVER transcripts under the round-1 rule) and C1 + C2 (ROVER under `rasm`). Each also reports one-edit, one-word and shifted flagged, and repeat misses.
- The bar is `passes_bar`: at least 38/40 agreement, one-edit and one-word flagged ≥ 90%, no repeat misses.
- If something passes, it becomes a pre-screen option. The default changes only if it clearly beats `nemotron-likelihood` on the bar.

### B. Prompt ablation (preregistered rule, in-sample windows)

Round-1 rule, the same 40 windows and controls. "Flagged" means the control was called mismatch or unsure. Agreement ranges are Wilson 95%.

| Prompt | "Quran" | "repetitions and mistakes" | Agreement | One-edit flagged | One-word flagged | Shifted flagged | Repeat misses | One-edit misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `verbatim` (model card, the round-1 decision) | – | – | 23/40 (42–71%) | 38/40 | 29/31 | 40/40 | 0 | `p15-x8`, `e19-x12` |
| `recitation` | yes | yes | **26/40** (50–78%) | **40/40** | **31/31** | 40/40 | 0 | – |
| `mistakes` (asked for) | no | yes | 24/40 (45–74%) | 39/40 | 30/31 | 40/40 | 0 | `p15-x8` |
| `quran` | yes | no | 26/40 (50–78%) | 39/40 | 30/31 | 40/40 | 0 | `p15-x8` |
| `plain` | no | no | 25/40 (47–76%) | 39/40 | 30/31 | 40/40 | 0 | `p15-x8` |

- **Dropping "Quran" does not help.** `mistakes` loses 2 net against `recitation`: it gains `p10-x6` but loses `e05-x7` (the restart collapsed again), `e08-x1` and `e19-x12`. Without the mistakes clause the pattern repeats: `plain` is 1 below `quran` (`e05-x7`).
- **The clause does nothing measurable.** `recitation` vs `quran` is +`e19-x12` / −`p10-x6`, and `mistakes` vs `plain` is −`e08-x1`. Both are within noise.
- **The real gain is leaving the model-card template.** Against `verbatim`, all four "Transcribe this recitation … exactly as recited" prompts recover `e13-x4` and `p12-x7`, and three of the four recover `e08-x1`. All four lose `p06-x1`, which `verbatim` matched. The net is +1 to +3.
- **`recitation`'s perfect one-edit score is luck.** On `p15-x8` the other three new prompts drop the planted word exactly as `verbatim` did. `recitation` only flags it because it hallucinated a sermon after the planted word (Part A).
- **Prompt choice is per-window noise, not signal.** 21 windows match under all five prompts and 12 under none. An oracle picking the best prompt per window would reach 28/40.
- `pick_prompt` chose `recitation` for C2: the most agreement among the prompts with controls ≥ 90% and no repeat misses.

### C. Rasm comparison and ROVER

All rows use the same 40 windows. "C1" is the `rasm` rule, "C2" is ROVER over six perturbed `recitation` transcripts.

| Row | Status | Agreement | One-edit | One-word | Shifted | Repeat misses |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline: `recitation`, round-1 rule | exploratory in round 1 | 26/40 (50–78%) | 40/40 | 31/31 | 40/40 | 0 |
| **C1 alone:** `recitation`, `rasm` | preregistered | 25/40 (47–76%) | 40/40 | 31/31 | 40/40 | 0 |
| C1 alone: `verbatim`, `rasm` | preregistered | 23/40 (42–71%) | 39/40 | 30/31 | 40/40 | 0 |
| C1 alone: Qwen3-ASR, `rasm` | preregistered | 5/40 (was 8) | 39/40 | 31/31 | 40/40 | **1 (`p05-x6`, new)** |
| C1 alone: Whisper, `rasm` | preregistered | 5/40 (was 7) | 36/40 | 31/31 | 40/40 | 4 (as round 1) |
| **C2 alone:** ROVER, round-1 rule | preregistered | **27/40** (52–80%) | **40/40** | **31/31** | 40/40 | 0 |
| **C1 + C2:** ROVER, `rasm` | preregistered | 26/40 (50–78%) | 40/40 | 31/31 | 40/40 | 0 |
| ROVER + Qwen3 + Whisper voters, round-1 or `rasm` | preregistered variant | 26/40 | 40/40 | 31/31 | 40/40 | 0 |
| ROVER, `rasm-no-edge` / `rasm-no-hamza` / `rasm-phonetic` | sensitivity / sensitivity / exploratory | 25 / 27 / 26 | 40/40 | 31/31 | 40/40 | 0 |

The six ROVER voters on their own, `recitation` prompt, round-1 rule:

| Voter | Raw | Denoised (DeepFilterNet3) |
| --- | --- | --- |
| 1.0× | 26/40 (this is the cached `recitation` transcript) | 21/40 |
| 0.9× | 24/40 (one-edit miss on `e11-x11`) | 19/40 |
| 1.1× | 25/40 | 20/40 |

Every voter flags all shifted controls and has no repeat misses. In 29 of the 40 windows at least one voter matches, which is the ceiling for any vote over these six.

**Nothing passes the bar.** The best row is ROVER under the round-1 rule: 27/40 with every control flagged, 11 windows short of 38.

#### Why C1 (rasm) loses

The skeleton fixed exactly the two windows Part A predicted, and only with hamza dropped:
- `e20-x6`: `كلا علام` merges to `كلعلم`;
- `p03-studio-5`: the session context absorbs the neighbour bleed.

It loses as many elsewhere, for two reasons:

- **Short skeletons make one-letter errors heavier.** With the alifs gone, `مهادا` / `كهادا` becomes `مهد` / `كهد`, 0.67 instead of 0.8, now doubtful (`p12-x7`). In the same way `إنا` / `إننا` is now 0.67 (`e19-x12`). Similarity thresholds tuned on full spellings do not carry over to skeletons. Some of those newly doubtful words were really wrong, for example Qwen and Whisper writing `آلام` (pains) for `آلاء` (bounties), which the round-1 rule accepted at 0.75.
- **Context words can absorb an unrelated heard word.** On Qwen `p05-x6` the repeat-removed control matches under `rasm`: the edge neighbour `مَعَكُمْ` pairs with a stray heard `آمنوا`, and context words never count. That is a repeat miss the round-1 rule did not have, and it disappears with the edge part off (`rasm-no-edge`). A fix would be to let context words absorb only heard words that match them (sim ≥ 0.75). Making that fix after seeing this result would be tuning on the test, so it was not run.

Phonetic merging changed no Nemotron window. `p05-x6` still fails on `قلوب` → `قلوبهم` even once `فذبطوا` matches.

#### What C2 (ROVER) does

Net +1 against the single `recitation` run.

What it fixed or improved:
- It fixed `p10-x6`: five of six voters heard `فَانٍ وَيَبْقَى` and outvoted the single run's `فَالْوَاقِعُ`.
- It moved `e20-x10` from mismatch to unsure.

What it held, sometimes narrowly:
- It held `e05-x7` only on a 3–3 tie between the raw voters, which kept the broken first attempt, and the denoised ones, which collapsed it into the canonical 26:109.
- It held `e08-x1` (`لآية` 2 against `لآيتهم` 2 and `لا آتاه` 2).

Nothing it changed let a control through.

- **Denoising hurts.** DeepFilterNet3 costs 5 to 7 matches at every speed. On clean studio windows it pushes the model toward the canonical text: at 1.0× the denoised run loses `e05-x7`, `e08-x1`, `e19-x12`, `e19-x9`, `e12-x3`, `p14-x2` and `p09-studio-317`. On phone copies it causes hallucinations (`p06-phone-173` became `فَإِذَا بَلَغَ الْحُيُولَ …`). At 1.0× it gained only `p10-x6` and `p03-studio-5`. As a result half the voters are worse than the pivot, and the vote mostly restores the raw run rather than improving on it.
- **Adding Qwen3 and Whisper as voters is a wash (26/40).**
  - Gains: `p06-x1`, where they outvote the hallucinated `ضل عن سبيلهم` loop, and `p03-studio-5`.
  - Losses: `p14-x2`, `e08-x1` and `e05-x7`, where their edge drops and collapsed restarts tip close votes.
- **Errors are correlated.** ROVER only removes errors that differ between perturbations. Nemotron's errors in this set are mostly systematic: the same consonant confusions on both copies of a repeat, the same canonical prior at the planted word, the same edge behaviour. Speed and denoising do not decorrelate them. That is why six voters gain one window.

### Decision

- **Nothing passes, so nothing new is wired.**
  - The prompt ablation tops out at 26/40.
  - C1 at 25/40.
  - C2 at 27/40 (52–80%).
  - C1 + C2 at 26/40.
  - Every row keeps its controls, but agreement stays 11 or more windows short of 38/40.
- **The default stays `nemotron-likelihood`.** It still clears 5 of 40 windows on `v1-check`, so the listener hears 35 of 40.
  - For comparison, ROVER as a pre-screen would keep 28 of 40: its 13 disagreements, 13 more splices for the seam listen, and a 2-window spot check.
  - That would save the listener 7 more windows than `nemotron-likelihood` does. But at 27/40 agreement, the cleared windows rest on an opinion that is wrong about a third of the time on correct audio. The bar exists to stop exactly that.
- **What would change it (speculative):**
  - The errors left over are mishearings at the planted word and canonical-prior collapses. Neither comparison nor voting fixes those.
  - The remaining lever is not free transcription. One option is to constrain decoding to the labelled text and its neighbours, which is what arm 2 does. Another is a model fine-tuned on non-canonical recitation, so that its prior stops pulling toward the expected verse.

### Cost and records

- **GPU and cost:**
  - Modal H100: about 5 min for B (120 transcriptions) and about 7.5 min for the round-2 app. Of those 7.5 min, 3.5 min was the 200 perturbed transcriptions and the rest was the H100 load and CPU stages.
  - Denoising ran on CPU (8 cores) and took a few minutes.
  - Roughly $1 in total (estimate).
- **Records** on the `tilawa-lab` volume:
  - `/lab/reviews/lab-v1.1/asr-nemotron-{mistakes,quran,plain}-prompt.json`: B, per item;
  - `asr-nemotron-rover.json`: all six voters per item with prompt, speed, denoise flag and audio sha256, plus the ROVER and ROVER + Qwen + Whisper outputs;
  - `asr-calibration.nemotron-prompt-ablation.json` and `asr-calibration.round2.json`: every source × rule, with stats and per-item tables;
  - `rover/denoised/*.wav`: the 40 denoised windows, the inputs behind the recorded hashes.
- **Tables:** `docs/benchmarks/lab-review-asr-v1-nemotron-prompt-ablation.json`, `docs/benchmarks/lab-review-asr-v1-round2.json` and `docs/benchmarks/lab-review-asr-v1-errors.json` (Part A).
- **Code:** `rasm`, `compare_rasm`, `edge_context`, `judge_rasm`, `rover`, `calib_stats`, `passes_bar` and `pick_prompt` in `tools/review_asr_core.py`, with tests in `tools/test_review_asr.py`. The run itself is `modal run modal/review_asr.py::round2_main`.

## Commercial APIs: Soniox and Speechmatics (2026-10-29)

A bake-off between two commercial ASR APIs as the second independent model of two-model triage (ticket 17), alongside Nemotron's likelihood arm. The winner then re-triages lab-v1.3. Code: `modal/review_api.py` (calls), `tools/review_api_core.py` (readings, scoring, winner), `tools/lab_triage_core.py` (the triage rule, now taking any second model), tests in `tools/test_review_api.py`.

### Preregistration (committed before any API output)

**Vendors and requests.** Each vendor is called only from a Modal function that mounts its own secret. Each request carries only the window's 16 kHz mono PCM16 WAV, the same bytes and sha256 every other model heard (`review_asr.clip_bytes`), named by its sha256, plus the language. It never carries the text, the label type, a context or a vocabulary.

- **Soniox**, async API: model `stt-async-v5` (its current async model; `stt-async-v4` is an alias), `language_hints: ["ar"]`, and `language_hints_strict: true` if `GET /v1/models` says the model supports it. The file and the transcription are deleted after the transcript is fetched. List price $0.10 per audio hour.
- **Speechmatics**, batch API v2 (`eu1`): `language: "ar"`, `model: "enhanced"` (the most accurate model; `model` replaces the deprecated `operating_point`). The job is deleted after the json-v2 transcript is fetched. List price $0.75 per audio hour (Enhanced, pre-recorded, pay as you go).
- Records per test set: `/lab/reviews/<testSet>/asr-<vendor>.json`, with the request config, per item the audio sha256, raw responses, transcript, timing (upload, submit to done, total) and estimated cost. No keys.

**Ground truth: your listening verdicts.** Every verdict in `/lab/reviews/{lab-v1.1,lab-v1.2,lab-v1.3}/me.json`, on the item exactly as you heard it (from the queue the verdict names).
- `ok` means the label fits; `bad_splice` (or any bad-data verdict) means it doesn't.
- Byte-identical clips with the same text (carried over between test sets) count once, keeping the earliest.
- Expected: about 150 items, 10 of them bad (3 on lab-v1.1, 7 on lab-v1.2).
- The set is not a random sample. Most lab-v1.2 items are ones the Nemotron + FastConformer triage did not auto-accept, so it is enriched for hard cases.

**Readings of one transcript**, all fixed here:
- **judge (primary; decides pairing and triage):** `review_api_core.judge`. Both the labelled (claimed) text and the window's alternative text go to the space-free rasm skeleton (`core.rasm`, hamza kept).
  - The alternative is the same one lab triage scored: the edit undone, the repeat removed, or a seeded one-word change for clean windows.
  - The distance from the transcript to each text is the letter edit distance to its best-fitting contiguous stretch. The text's ends are free, so edge words the model didn't write cost nothing.
  - gap = d(alternative) − d(claimed), and need = max(2, half the distance between the two texts).
  - match if gap ≥ need, mismatch if gap ≤ −need, unsure otherwise.
  - Also unsure: an empty transcript, one more than 50% of its own length away from the nearer text, or one shorter than 50% of the nearer text.
  - The comparison runs locally after the transcript is saved. The API never sees either text.
- **opinion:** the round-1 rule `core.compare` against the labelled text alone (the ticket 12 decision rule).
- **opinionRasm:** `core.compare_rasm` with no edge context.

**Gauge**, per API:
1. Agreement: match on your `ok` items.
2. Sensitivity on your bad items: flagged (not match) and called mismatch. This is the most important number.
3. Ticket 12's synthetic controls on the 40 lab-v1.1 v1-check windows (`core.controls`, seed 12): one-edit, one-word and shifted flagged, for comparability with the earlier models. Under the judge, a control text is the claimed side and the labelled text the alternative. Also: the edit-undone control on every edited window you heard as ok.
4. Repeat and restart misses: the text without the repeat is accepted on a repeat or restart you heard as ok.
5. Failure taxonomy of the opinion's rejections on `ok` items, using round 2's automatic buckets in order:
   - hallucination: boilerplate, or more than 3 extra words;
   - canonicalisation: the edit-undone text matches the transcript;
   - orthography: equal skeletons, or the rasm rule matches;
   - edge word: only edge reasons;
   - otherwise mishearing.

   Each is tagged at the edit (the planted word or a neighbour) and phone copy.
6. Cost (list price × audio duration) and latency (submit to done, median and p90).

**Pairing with Nemotron**, on your verdicts:
- Nemotron's decision is its lab triage likelihood decision on the same window and text (`triage-v1.json`); otherwise it is unsure.
- Both match: auto-accepted. Anything else reaches you.
- Counted: your `ok` items auto-accepted, your bad items auto-accepted (must be about 0), and the number of items queued.
- The old Nemotron + FastConformer pairing is reported on the same items.

**Winner** (`pick_winner`), the API whose Nemotron pairing under the judge has, in order:
1. the fewest of your bad items auto-accepted;
2. the most of your `ok` items auto-accepted;
3. the most bad items flagged by the judge;
4. the most one-edit controls flagged by the judge;
5. the lower list price.

**Small n.** There are only 10 bad items. 0/10 auto-accepted still leaves a 95% upper bound of 28% on the true rate. Differences of 2 or fewer items are noise (spec decision 12). Ranges are 95% Wilson intervals.

**Re-triage of lab-v1.3** with Nemotron + the winner, using the lab triage rule unchanged:
- Inputs: the same 1700 cases, windows and alternatives as `triage-v1.json`. Nemotron's decisions are reused as they are; the winner's judge replaces FastConformer.
- Outcomes and spot checks are as before: 15 + 15 random, and seam checks of 10% capped at 20, seed 17.
- Outputs: `review/triage-<winner>.json` and `review/triage-<winner>-confirm.json`. `triage.json` is untouched.
- Your lab-v1.3 verdicts are keyed by id, so they show on the same clips.
