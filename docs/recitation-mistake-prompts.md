# Research-informed recitation recording prompts

Research date: 2026-09-30. Scope: the volunteer collection at `/help`, not a model retrain or a change to the SDK's mistake detector.

## What the evidence supports

The previous generator selected unrelated replacement words and randomly changed pronounceable short vowels. This makes labels convenient, but often produces errors without a plausible reading or memorization mechanism.

- **Articulation, heavy/light consonants, nasalization and lengthening matter.** Hassan & Zailaini studied recorded recitations from 20 IIUM level-2 learners. They identified these categories alongside letter/vowel and stopping errors. Their detailed tables include insufficient nasalization on doubled noon/meem, shortened madd lazim, and vowel/sukun/shaddah mistakes. This is evidence for those mechanisms, not worldwide incidence. The passages were selected to exercise tajweed rules, and the analysis excluded errors below a 1% threshold. [Original paper, 2013, DOI 10.1016/j.sbspro.2013.10.318, especially pp. 140–144](https://darulquran.co.uk/wp-content/uploads/2021/02/Analysis-of-Tajweed-Errors-in-Quranic-Recitation.pdf).
- **Similar passages challenge memorization.** Interviews with four tahfiz lecturers and 30 students identify mutashabihat verses as a significant memorization difficulty. This supports mixing up words in parallel passages rather than inserting an unrelated word. It does not publish a population rate for each mistake. [Ahmad et al., 2022, university repository](https://oarep.usim.edu.my/entities/publication/08f3b4d2-1bb0-45ba-aa67-68172cdcf317).
- **Qaf/kaf confusion has concrete learner evidence.** A small observational study of two 12-year-old Al-Falaq learners describes qaf being pronounced as kaf. The sample is far too small to rank this among all learners' mistakes. [Falya Fadia, 2024, UPI thesis repository](https://repository.upi.edu/124891/).
- **Letter/vowel changes differ from subtle timing errors.** The Al-Otrojah teaching curriculum describes letter substitution and short-vowel/sukun errors separately from inadequate ghunna/madd. [School curriculum, pp. 5–6](https://schools.macnet.ca/alotrojah/wp-content/uploads/sites/10/2019/10/02-Introduction-to-Tajweed-Part-1-Unit-2-Chapter-1.pdf).
- **Real learner audio and expert review remain necessary.** Salameh et al. collected more than 7,000 recordings from 1,287 non-Arabic speakers with expert-reviewed labels. Their six recording categories concern correctness and recording validity; they are not a frequency distribution over our seven mistake classes. [Authors' dataset paper, 2024](https://arxiv.org/html/2405.02675v1).

## What changed

The v2 generator keeps the existing classes and their coverage weights. Those weights are sampling quotas, **not estimates of real-world frequency**. It samples a bank of 33 explicit scenarios with stable IDs and mechanisms, using the repository's Uthmani text rather than hand-transcribed whole verses.

| Existing class | Mechanism and illustrative prompt |
| --- | --- |
| substitution | Parallel-passage interference: 7:111 `وَأَرْسِلْ` → 26:36 `وَٱبْعَثْ`; 12:2 `أَنزَلْنَٰهُ` → 43:3 `جَعَلْنَٰهُ`; neighbouring prophet names in Surah 26; 79:39/41 `ٱلْجَحِيمَ`/`ٱلْجَنَّةَ`. Also qaf → kaf in Al-Falaq. |
| skip_word | Omit the second `وَإِيَّاكَ` in 1:5, or a short negation/grammatical word. |
| skip_ayah | Lose position between repeated `الناس` endings, neighbouring parallel verses in Al-Kafirun, or the repeated hardship/ease wording in Ash-Sharh. |
| repeat | Brief hesitation and a repeated word. Stored as a fluency control, not a claim that every restart violates tajweed. |
| vowel | A specified audible letter/vowel change. For 1:7 `أَنْعَمْتَ` → `أَنْعَمْتُ`, explicitly continue into `عَلَيْهِمْ`; stopping would hide the change. |
| tajweed | Shorten internal madd lazim in `ٱلضَّآلِّينَ` to two instead of six counts; omit sustained doubled-noon/meem nasalization while retaining the doubling; read sad too lightly; omit shadda in `ٱلْخَنَّاسِ`. Metadata separates gemination loss from timing/emphasis errors. |
| none | Normal recitation across the broader corpus. |

These are **constructed examples of supported mechanisms**, not a claim that a cited study observed each particular word/verse mutation. The omission and hesitation locations are design choices informed by repetition and fluency, not source-backed frequency rankings. The same applies to each chosen vowel site and parallel word pair. The new scenarios still need review by a qualified Hafs recitation teacher before they are treated as a gold-standard acoustic evaluation set.

Valid stopping/restarting and transmitted reading variants must not be counted as mistakes merely because their acoustic realization differs. We avoid random verse-final vowel edits and variable final madd duration; the selected shortened madd is internal lazim. Do not treat normal-control recordings as expert-certified correct without listening to them. A pronounced self-correction also needs a separate temporal annotation; the existing `extra_mistake` flag is not enough for that.

## Label integrity and compatibility

- New prompts are version 2; `scenario`, `mechanism`, `elicitation`, and `fluency_control` are saved with each submission. This makes narrow scenario coverage visible in exports and permits stratified evaluation.
- The server signs the full issued label, and the client returns that token. Submission verifies it instead of regenerating from the current bank. This also preserves an explicitly requested `?kind=` prompt.
- The original generator is frozen as `prompts-v1.mjs` for pending legacy clients and historical seed replay. Existing recorded labels and audio are not rewritten.
- `HELP_PROMPT_SECRET` should be a stable private key if in-progress v2 prompts need to survive process restarts or requests are served by multiple processes. Without it, a random process-local key is used; a pending token from an earlier/different process is rejected rather than mislabelled. Restarting the recording flow obtains a new prompt.
- Prompt-derived labels describe the requested performance. A contributor's confirmation does not prove that the audio contains exactly that error. Expert acoustic verification is still required.

## How to measure actual representativeness next

Collect consented natural learner recitations alongside acted scenarios; annotate them with a qualified teacher, including self-correction, legitimate stops/restarts, reader level and language background. Estimate occurrence within each reader group, not across a mixed convenience sample. Evaluate on speaker-disjoint data and report results separately for natural errors, acted scenarios and fluency controls. A speaker-disjoint split alone is insufficient with a small curated bank: reserve unseen passages/scenarios too, so the detector cannot pass by memorizing prompts.

Do not use the study's small-cohort percentages as universal training weights. Expand the bank after teacher review and real-audio error analysis, especially for ikhfa/idgham, sukun, word endings in continuous reading, phrase-level omission and recovery. The present change improves collection prompts; it does not establish that the resulting dataset is statistically representative or that model accuracy improved.
