"""Check a locally recreated teacher graph against the original fp32 head.

Uses synthetic features and available training-only audio, never practice labels.
The unchanged teacher export has a build-dependent graph serialization.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from quran_muaalem import Muaalem
from transformers import SeamlessM4TFeatureExtractor


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    root = Path(os.environ['TEACHER_MODEL'])
    teacher = Muaalem('obadx/muaalem-model-v3_2', device='cpu', dtype=torch.float32)
    teacher.model.eval().float()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    graph = ort.InferenceSession(str(root/'teacher.onnx'), options, providers=['CPUExecutionProvider'])
    cases = [(f'random-{n}', torch.randn(1, n, 160), torch.ones(1, n, dtype=torch.int64)) for n in (24, 149, 300)]
    extractor = SeamlessM4TFeatureExtractor.from_pretrained('obadx/muaalem-model-v3_2')
    audio = next(p for p in sorted((Path(os.environ['LAB_DATA'])/'help-train').rglob('*'))
                 if p.suffix.lower() in ('.wav', '.flac'))
    wave, rate = sf.read(audio, dtype='float32')
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if rate != 16000:
        import torchaudio
        wave = torchaudio.functional.resample(torch.from_numpy(wave), rate, 16000).numpy()
    features = extractor(wave[:8*16000], sampling_rate=16000, return_tensors='pt')
    cases.append(('training-audio', features.input_features, features.attention_mask.to(torch.int64)))
    results = []
    for name, x, mask in cases:
        with torch.no_grad():
            expected = teacher.model(input_features=x, attention_mask=mask, return_dict=False)[0]['phonemes'].log_softmax(-1).numpy()
        actual = graph.run(None, {'input_features': x.numpy(), 'attention_mask': mask.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
        row = dict(case=name, frames=int(actual.shape[1]),
                   max_probability_difference=float(np.abs(np.exp(actual)-np.exp(expected)).max()),
                   top1_agreement=float((actual.argmax(-1)==expected.argmax(-1)).mean()))
        assert row['max_probability_difference'] < 2e-5
        assert row['top1_agreement'] == 1
        results.append(row)
    files = [dict(file=p.name, bytes=p.stat().st_size, sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest())
             for p in (root/'teacher.onnx', root/'teacher.onnx.data')]
    report = dict(cases=results, files=files, training_audio=str(audio), training_audio_sha256=hashlib.file_digest(audio.open('rb'), 'sha256').hexdigest())
    (Path(os.environ['LAB_OUT'])/'teacher-parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
