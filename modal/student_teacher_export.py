"""Export the existing fp32 KD teacher's phoneme head for the SDK seam ear."""
import os
from pathlib import Path

import torch
from quran_muaalem import Muaalem

torch.set_num_threads(2)
root = Path(os.environ['TEACHER_OUT'])
root.mkdir(parents=True, exist_ok=True)
teacher = Muaalem('obadx/muaalem-model-v3_2', device='cpu', dtype=torch.float32)
teacher.model.eval().float()

class PhonemeHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, attention_mask):
        logits = self.model(input_features=input_features, attention_mask=attention_mask, return_dict=False)[0]['phonemes']
        return logits.log_softmax(-1)

model = PhonemeHead(teacher.model).eval()
frames = torch.export.Dim('frames', min=8)
args = (torch.zeros(1, 300, 160), torch.ones(1, 300, dtype=torch.int64))
torch.onnx.export(model, args, str(root / 'teacher.onnx'),
    input_names=['input_features', 'attention_mask'], output_names=['log_probs'],
    dynamic_shapes=({1: frames}, {1: frames}), opset_version=18, dynamo=True,
    external_data=True, optimize=False, report=True, artifacts_dir=str(root))
print('exported', [(f.name, f.stat().st_size) for f in root.iterdir()], flush=True)
