"""The actual offline export must not freeze Zipformer's positional cache."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import student_train  # installs the pinned Zipformer source path
from zipformer import CompactRelPositionalEncoding
from student_offline import export_graph

class Export(unittest.TestCase):
    def test_long_inputs_extend_positions_in_the_serialized_graph(self):
        import onnxruntime as ort
        class Positions(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.positions = CompactRelPositionalEncoding(16, 0.0)
            def forward(self, x, lens):
                n = x.shape[1]
                return self.positions(x.transpose(0, 1)).reshape(1, 2*n-1, 4, 4), lens
        model=Positions().eval()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"positions.onnx"
            export_graph(model, torch.zeros(1, 401, 80), torch.tensor([401]), path)
            session=ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for n in (401, 1999, 2001, 3201, 6401):
                x=torch.zeros(1,n,80); lens=torch.tensor([n])
                with torch.no_grad(): expected,_=model(x,lens)
                got,_=session.run(None, {"x":x.numpy(), "x_lens":lens.numpy()})
                self.assertEqual(got.shape, (1,2*n-1,4,4))
                np.testing.assert_allclose(got,expected.numpy(),atol=2e-6,rtol=2e-6)

if __name__=="__main__": unittest.main()
