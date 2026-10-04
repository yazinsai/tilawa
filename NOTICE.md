# Notices

This repository is MIT-licensed (`LICENSE`). Some vendored models, lexicons, and scripts are **not**.

## Acknowledgements

- **Quran-Lab** ([`Quran-Lab/zipformer_p-arabic-v3`](https://huggingface.co/Quran-Lab/zipformer_p-arabic-v3), author Muno459 / Quran-Lab): streaming Zipformer2-CTC acoustic model, 251-token vocabulary, phoneme lexicon (`quran_text2phoneme.json`), and eval/export scripts. NPL-1.2 §6 does not require attribution; we credit them anyway. Full text: [`licenses/NPL-1.2.txt`](licenses/NPL-1.2.txt).
- **alketab** ([ملقّن القرآن](https://prompter.alketab.app/), `@alketab/quran-engine`): design source for the recitation pipeline specified in `lab/docs/specs/recitation-engine-spec.md`. The phoneme corpus in `lab/data/zipformer/quran.json` comes from alketab (itself derived from Quran-Lab's `quran_text2phoneme.json`). Nothing from the recovered alketab engine remains in this repository. `packages/core/src/recitation/` is a clean-room MIT reimplementation from that spec plus the dump-vector oracles in `lab/docs/specs/vectors/` (dumped from the original engine before its removal at `e172b79`). Licence of the original engine was unstated; we grant no licence to that recovered source.
- **k2 / icefall** (Apache-2.0) and **lhotse** (Apache-2.0): Zipformer training and export stack.
- **onnxruntime** (MIT): ONNX inference.
- Training audio for our fine-tunes: EveryAyah (`tarteel-ai/everyayah` / `greentechapps/everyayah_curated_1s_20s`, MIT); QUA (`hetchyy/quranic-universal-ayahs`, CC-BY-4.0); Iqra (`IqraEval/Iqra_train`, unstated); RetaSy (`RetaSy/quranic_audio_dataset`, unstated); TLOG (`tarteel-ai/tlog`, unstated). QuranTTS (`Quran-Lab/QuranTTS`, NPL-1.2) is excluded. See `lab/docs/plans/2026-09-14-sota-tilawa.md` §3.1 and `lab/scripts/prepare_zipformer_data_modal.py`.

## Licensing of model artefacts

These artefacts are **NPL-1.2 Derivatives** of Quran-Lab's Work. NPL-1.2 §7 is share-alike: a Derivative includes models trained, fine-tuned, or *evaluated* with the Work, and datasets, lexicons, or label sets produced from it. They are **not** covered by this repository's MIT licence:

- `lab/data/zipformer/quran_phoneme_zipformer.onnx` (byte-identical to Quran-Lab `zipformer_p_arabic_v3.1.int8.onnx`)
- Fine-tuned checkpoints (`ft-*`)
- Blended model `interp-gentle-a0.5` (0.5 v3.1 + 0.5 ft-gentle), the previous default
- Default model `a0w-ep1-a0.5` (`zipformer_a0w_ep1_a05.int8.onnx`: 0.5 v3 + 0.5 a0w-w2 epoch 1, fine-tuned on the training audio above; int8, with an extra encoder-frame output), distributed with `NPL-1.2.txt` at the `zipformer-a0w-ep1-a0.5` GitHub release
- 251-token vocabulary `lab/experiments/zipformer-ctc/tokens.txt`
- Phoneme lexicon in `lab/data/zipformer/quran.json`
- Training labels produced from that lexicon
- Copied eval scripts in `lab/experiments/zipformer-ctc/reference_tools/` (already marked NPL-1.2)

NPL-1.2 §§3/5/9: you may not charge for the Work or any feature it powers; hosted use is free or cost recovery only. Every Derivative must be distributed under NPL-1.2 (or a later Quran-Lab version). See [`licenses/NPL-1.2.txt`](licenses/NPL-1.2.txt).

Repository code and non-derived assets remain MIT.
