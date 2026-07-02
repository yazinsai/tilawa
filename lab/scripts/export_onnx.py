"""Export FastConformer to ONNX (CTC-only) and extract BPE vocabulary."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_onnx():
    import nemo.collections.asr as nemo_asr

    model_id = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    out_dir = PROJECT_ROOT / "data"
    onnx_path = out_dir / "fastconformer_ar_ctc.onnx"
    vocab_path = out_dir / "vocab.json"

    print(f"Loading model: {model_id}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    model.eval()

    # Export CTC-only (single ONNX file)
    model.set_export_config({"decoder_type": "ctc"})
    print(f"Exporting to: {onnx_path}")
    model.export(str(onnx_path))
    print(f"ONNX export complete: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Extract BPE vocabulary
    tokenizer = model.tokenizer
    vocab = {}
    for i in range(tokenizer.vocab_size):
        vocab[str(i)] = tokenizer.ids_to_tokens([i])[0]
    # Add blank token (last index for CTC)
    vocab[str(tokenizer.vocab_size)] = "<blank>"

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved: {vocab_path} ({len(vocab)} tokens)")


def validate_onnx():
    """Validate ONNX model produces same output as NeMo native."""
    import numpy as np
    import onnxruntime as ort
    import nemo.collections.asr as nemo_asr
    import soundfile as sf

    model_id = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    onnx_path = PROJECT_ROOT / "data" / "fastconformer_ar_ctc.onnx"

    # Load NeMo model
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    model.eval()
    model.change_decoding_strategy(decoder_type="ctc")

    # Use a test audio file
    test_files = sorted((PROJECT_ROOT / "benchmark" / "test_corpus").glob("*.wav"))
    if not test_files:
        print("No test audio files found. Skipping validation.")
        return
    test_audio = test_files[0]
    print(f"Validating with: {test_audio.name}")

    # NeMo native transcription
    outputs_nemo = model.transcribe(
        audio=[str(test_audio)], batch_size=1, return_hypotheses=False, verbose=False
    )
    if isinstance(outputs_nemo, tuple):
        outputs_nemo = outputs_nemo[0]
    raw_out = outputs_nemo[0] if isinstance(outputs_nemo, list) else outputs_nemo
    nemo_text = raw_out.text if hasattr(raw_out, "text") else str(raw_out)
    print(f"NeMo output:  {nemo_text[:100]}")

    # ONNX inference with NeMo's own preprocessor
    import torch
    audio_data, sr = sf.read(str(test_audio), dtype="float32")
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
    audio_tensor = torch.tensor(audio_data).unsqueeze(0)
    audio_length = torch.tensor([audio_tensor.shape[1]])

    # Use NeMo's preprocessor
    processed, proc_len = model.preprocessor(
        input_signal=audio_tensor, length=audio_length
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f"ONNX inputs:  {[(n, sess.get_inputs()[i].shape) for i, n in enumerate(input_names)]}")
    print(f"ONNX outputs: {[(n, sess.get_outputs()[i].shape) for i, n in enumerate(output_names)]}")

    ort_inputs = {
        input_names[0]: processed.detach().numpy(),
        input_names[1]: proc_len.detach().numpy(),
    }
    logprobs = sess.run(None, ort_inputs)[0]

    # CTC greedy decode
    vocab_path = PROJECT_ROOT / "data" / "vocab.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    blank_id = max(int(k) for k in vocab.keys())

    predicted_ids = np.argmax(logprobs[0], axis=-1)
    tokens = []
    prev = -1
    for idx in predicted_ids:
        if idx != prev and idx != blank_id:
            tokens.append(vocab.get(str(idx), "?"))
        prev = idx
    onnx_text = "".join(tokens).replace("\u2581", " ").strip()
    print(f"ONNX output:  {onnx_text[:100]}")

    # Compare
    if nemo_text.strip() == onnx_text.strip():
        print("MATCH: ONNX output matches NeMo native.")
    else:
        print("MISMATCH: Outputs differ. Investigate preprocessing.")
        print(f"  NeMo: {nemo_text}")
        print(f"  ONNX: {onnx_text}")


def quantize():
    """Quantize the ONNX model to uint8 for smaller download size."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    input_path = PROJECT_ROOT / "data" / "fastconformer_ar_ctc.onnx"
    output_path = PROJECT_ROOT / "data" / "fastconformer_ar_ctc_q8.onnx"

    if not input_path.exists():
        print(f"ONNX model not found: {input_path}")
        return

    print(f"Quantizing: {input_path} ({input_path.stat().st_size / 1e6:.1f} MB)")
    quantize_dynamic(str(input_path), str(output_path), weight_type=QuantType.QUInt8)
    print(f"Quantized:  {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate_onnx()
    elif "--quantize" in sys.argv:
        quantize()
    else:
        export_onnx()
        quantize()
        if "--skip-validate" not in sys.argv:
            validate_onnx()
