# General Arabic adaptation and Arabic Zipformer checkpoint search

Checked 2026-10-07. Research only: no audio download, model import, training, or application changes. Ranking emphasizes technical scale and quality; terms are recorded separately for later resolution.

## Recommendation

Start with filtered MGB-2 or QASR as the broad Arabic core, Common Voice as a smaller source of recording diversity, and then Quran adaptation. MASC is the next large candidate. General Arabic can teach Arabic acoustics and robustness; ordinary undiacritized transcripts do not provide reliable supervision for spoken short vowels or Quran mistakes. Improvement is a hypothesis requiring a matched-budget comparison.

The supplied current baseline is about 225k tlog clips / 535 hours. On 388 help clips from 29 speakers, English-pretrained plus soft5 produced 54/64 hits and 12 false positives (81.8% precision, 84.4% recall); teacher produced 57 hits and 17 false positives (77.0%, 89.1%). Vowel hits were 13/30 versus teacher 19/30. These are the relevant acceptance outcomes, alongside phone latency.

## Ranked data candidates

| Rank | Candidate and scale | Technical value and limitations | Access / known terms |
|---|---|---|---|
| 1 | **QASR: 2,041 hours, ~1.6M segments, 3,545 broadcast episodes. MGB-2: ~1,200 usable hours.** | Al Jazeera speech, mostly MSA with dialects; broad speakers and recording conditions. QASR has segment alignment quality scores and speaker metadata. Captions are lightly supervised and can omit repetitions or corrections. Filter for close audio/text agreement. These collections overlap: do not add their hours as independent data. MGB-2 has an existing icefall recipe. | QASR audio request form; card states noncommercial use. MGB-2 corpus agreement has commercial-use restrictions. |
| 2 | **MASC: 1,000 hours, over 700 YouTube channels; one mirror exposes ~913,400 rows.** | Diverse MSA/dialects and clean/noisy material at 16 kHz. Source is manually supplied closed captions, not uniformly verbatim annotation; mismatch, overlap, noise and alignment problems remain. Authors manually annotate 40 hours for evaluation. Training text removes diacritics and normalizes letters. Channels are not unique speakers. Split by source video/channel and check speaker leakage. | Public mirrors; CC-BY-4.0 tag on mirror. Underlying recording rights not independently verified. |
| 3 | **Common Voice Arabic 27.0: 158.27 recorded hours, 93.34 validated hours, 136,832 clips, 1,679 speakers.** | Crowd-recorded read speech adds speaker/device variety; no verified phone-recording fraction. Validation is useful but intended prompts are not detailed as-spoken annotations. 93.34 hours is validated total, not official training duration. Official train/dev/test contain 29,584 / 10,271 / 10,606 clips, while all validated clips total 80,696. Preserve held-out speakers if rebuilding splits. | Official Mozilla Data Collective portal; CC0 with additional access terms including no speaker reidentification or redistribution. Login/download not tested. |
| 4 | **SADA: roughly 667–668 hours; mirror ~253k segments.** | Saudi TV, useful Gulf/Najdi/Hijazi and MSA coverage. Includes transcript, dialect, age/gender and environment fields. Episode-local speaker labels do not establish global speaker independence. Exact dialect hours and phone share unverified. Strong regional supplement. | Official Kaggle release; mirror reports CC-BY-NC-SA-4.0. |
| 5 | **Casablanca: 48 hours across eight Arabic dialects.** | Smaller but useful human-transcribed dialect stress test, including code switching. Choose as evaluation or targeted supplement rather than the scale core. | Dataset card reports CC-BY-NC-ND-4.0. |

FLEURS is a useful separate read-speech evaluation source, not a large adaptation core: roughly 12 hours per language in the paper, with Arabic `ar_eg`; that code alone does not establish Egyptian colloquial speech. Exact current Arabic hours were not verified. ArVoice advertises 83.52 hours but about 73.5 hours are synthetic, leaving about 10 human hours / seven speakers; its diacritized text can support a targeted vowel experiment but is not broad real-speech scale.

### Sources

- [QASR original paper](https://arxiv.org/html/2106.13000v1), [QCRI dataset/access card](https://huggingface.co/datasets/QCRI/QASR).
- [MGB-2 original paper](https://people.csail.mit.edu/jrg/2016/Ali-SLT-16.pdf), [QCRI card](https://huggingface.co/datasets/QCRI/mgb2/blob/main/README.md), [icefall MGB-2 recipe](https://github.com/k2-fsa/icefall/blob/master/egs/mgb2/ASR/README.md).
- [MASC author-uploaded paper](https://www.researchgate.net/publication/367481673_MASC_Massive_Arabic_Speech_Corpus), [public mirror](https://huggingface.co/datasets/MohamedRashad/MASC-Arabic). Mirror metadata is weaker evidence than the original paper.
- [Common Voice 27.0 official Arabic release](https://mozilladatacollective.com/datasets/cmu5wom3z00ceo107guxtg4xa), released September 17, 2026.
- [SADA official release](https://www.kaggle.com/datasets/sdaiancai/sada2022), [mirror/card](https://huggingface.co/datasets/khaledalganem/sada2022). Official page did not expose readable contents in this search; detailed mirror claims need validation against release files before use.
- [Casablanca original dataset](https://huggingface.co/datasets/UBC-NLP/Casablanca), [FLEURS original dataset](https://huggingface.co/datasets/google/fleurs), [ArVoice original dataset](https://huggingface.co/datasets/MBZUAI/ArVoice).

## Explicit search: Arabic Zipformer besides Quran-Lab

| Candidate | Verified evidence | Fit for Tilawa |
|---|---|---|
| **PengChengStarling** | Genuine Zipformer encoder / RNN-Transducer, multilingual streaming model including Arabic. Authors say ~2,000 hours per language. Trainable `pretrained.pt` (1.32 GB) and streaming ONNX exports are published. Encoder ONNX is 1.15 GB. Author MGB-2 WER: 24.37 versus Whisper large-v3 30.3. Apache-2.0 model tag. | Verified fine-tuning candidate, but much larger than current 64.6M / 22.3M models. Exact parameter count not independently recovered; do not present ~330M as verified. No demonstrated Snapdragon 7xx performance. Multilingual subword transducer and encoder shapes require adaptation; not a direct phoneme-head weight replacement. Best considered an offline teacher or separate initialization experiment. |
| **Antares97/sau-zipformer-asr** | Card explicitly identifies Saudi Arabic Zipformer2 Transducer trained on GigaSpeech3 Arabic using icefall; recommended fixed-text-normalization model. Public tree contains ONNX directory, tokens and unigram model, total repository 529 MB. Apache-2.0 tag. | Promising independent Arabic model, but **no trainable checkpoint published in the inspected tree**. Parameter count, streaming cache interface, latency and evaluation metrics unverified. Ask author for `.pt`, exact training configuration and data manifest before calling it a fine-tuning base. SADA appears in another export filename; provenance needs checking. |

Sources: [PengChengStarling project](https://github.com/PCL-Voice/PengChengStarling), [model card](https://huggingface.co/stdo/PengChengStarling), [published files](https://huggingface.co/stdo/PengChengStarling/tree/main); [Saudi model card](https://huggingface.co/Antares97/sau-zipformer-asr), [published files](https://huggingface.co/Antares97/sau-zipformer-asr/tree/main).

The MGB-2 checkpoint named `icefall-asr-mgb2-conformer_ctc-2022-27-06` is a Conformer, so it is not evidence of Arabic Zipformer weights. Muno459 / Quran-Lab-related phoneme assets remain excluded under current PLAN constraints. The search found no verified independent Arabic checkpoint matching the current small listener shapes; this is not a claim that none exists.

## Proposed experiment, before any large run

1. Audit a few hundred candidate clips across sources for exact spoken-text agreement, single-speaker speech and retained Arabic sounds. Estimate usable hours after filtering rather than buying nominal hours. Deduplicate audio and source episodes against Quran data and every evaluation split.
2. Use a provisional pilot of 100–150 high-confidence MSA hours from QASR/MGB-2 plus 30–50 validated Common Voice hours, subject to actual available training hours after speaker exclusion. These are experiment budgets, not measured corpus yields.
3. Compare current English initialization directly against the same initialization plus Arabic adaptation, with equal Quran fine-tuning steps, data, seeds and scoring settings. A temporary Arabic text head is an option for encoder adaptation; restore/retrain the Quran phoneme head afterward. Do not manufacture short-vowel labels from undiacritized spelling and treat them as observed sound.
4. Judge help-set consonant confusions, vowel hits, false warnings and speaker-disjoint Quran phoneme error. Keep gold audio excluded from training. Report warning precision/recall together, and measure word-highlight p95 ≤600 ms and confirmation 2–3 seconds on the target 2020 Snapdragon 7xx / 4 GB phone. A broad-ASR WER improvement alone does not justify adoption.
5. If the pilot improves those outcomes, expand the filtered broadcast core and add MASC/SADA in separate ablations. Keep a matched-budget control so extra training alone is not mistaken for Arabic-data benefit. Add the required scoreboard row when any metric-changing experiment is actually run.

Related Quran sources and mistake-label discussion: [Quran data research](research-data-2026-10-07.md). Listener and confirmation recommendations: [listener research](research-listener-2026-10-07.md).
