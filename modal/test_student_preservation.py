"""Small-tensor checks of acoustic mistake preservation; no recording inputs."""

import unittest

import torch

from student_flat_basin import objective


class FixedFrames(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, feats, lens):
        return self.logits.log_softmax(-1)[None].expand(len(lens), -1, -1), lens


class MistakePreservationTest(unittest.TestCase):
    def test_preservation_keeps_shipped_deviant_phones_without_script_ctc(self):
        # The shipped ear hears class 2 while the candidate moved to class 1.
        # The recording's script is immaterial: preserve what shipped heard.
        model = FixedFrames([[0., 2., 0.]] * 4)
        anchor = FixedFrames([[0., 0., 2.]] * 4).requires_grad_(False)
        item = dict(record=dict(role="preserve", expected="unverified script"),
                    feats=torch.zeros(4, 2), target=None, teacher=None)
        cfg = dict(ctc_weight=1., kd_weight=0., kd_delay=0)
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        before = float(model.logits.detach()[0, 2])
        result, _ = objective(model, anchor, [item], cfg, torch.device("cpu"))
        self.assertEqual(float(result["ctc"].detach()), 0.)
        self.assertGreater(float(result["preservation"].detach()), 0.)
        result["loss"].backward()
        opt.step()
        self.assertGreater(float(model.logits.detach()[0, 2]), before)

    def test_rejects_accidental_script_transcript(self):
        model = FixedFrames([[0., 0., 2.]] * 4)
        item = dict(record=dict(role="preserve"), feats=torch.zeros(4, 2), target=[1], teacher=None)
        with self.assertRaisesRegex(ValueError, "verified transcript"):
            objective(model, model, [item], dict(ctc_weight=1., kd_weight=0., kd_delay=0), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
