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
