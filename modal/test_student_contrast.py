"""Small mathematical and matched-recording contracts; no benchmark inputs."""

import copy

import pytest
import torch

from student_contrast import (ContrastHead, admit_recordings, contrast_basis,
                              matched_refit, ordinary_controls, recording_loss,
                              speaker_partition)


def source():
    torch.manual_seed(91)
    return torch.nn.Linear(7, 5, dtype=torch.float64)


def test_basis_covers_every_phone_and_blank_without_a_common_direction():
    b = contrast_basis(43, dtype=torch.float64)
    torch.testing.assert_close(b.T @ b, torch.eye(42, dtype=b.dtype))
    torch.testing.assert_close(b @ b.T, torch.eye(43, dtype=b.dtype) - torch.ones(43, 43, dtype=b.dtype) / 43)
    torch.testing.assert_close(b.sum(0), torch.zeros(42, dtype=b.dtype), atol=1e-14, rtol=0)
    assert b[0].square().sum() > 0  # Blank participates in the contrasts.


def test_common_feature_shift_leaves_initial_probabilities_and_contrast_coordinates_identical():
    native = source()
    shifted = copy.deepcopy(native)
    with torch.no_grad():
        shifted.weight.add_(torch.arange(7, dtype=torch.float64)[None] * 11)
        shifted.bias.add_(37)
    a, b = ContrastHead(native), ContrastHead(shifted)
    torch.testing.assert_close(a.weight, b.weight, atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(a.bias, b.bias, atol=1e-13, rtol=1e-12)
    x = torch.randn(23, 7, dtype=torch.float64)
    torch.testing.assert_close(a(x).softmax(-1), native(x).softmax(-1))
    torch.testing.assert_close(b(x).softmax(-1), shifted(x).softmax(-1))


def test_folded_head_preserves_trained_consonant_versus_blank_preferences():
    head = ContrastHead(source())
    with torch.no_grad():
        head.weight.add_(torch.randn_like(head.weight) * .03)
        head.bias.add_(torch.randn_like(head.bias) * .02)
    x = torch.randn(17, 7, dtype=torch.float64)
    torch.testing.assert_close(head(x), head.folded()(x))
    assert head.weight.shape[0] == 4


def test_controls_are_functionally_equal_but_have_different_parameter_scales():
    native, x = source(), torch.randn(31, 7, dtype=torch.float64)
    controls = ordinary_controls(native, 7)
    for h in controls.values():
        torch.testing.assert_close(h(x).softmax(-1), native(x).softmax(-1))
    assert controls["shifted"].weight.square().mean() > controls["centered"].weight.square().mean()


def provenance(i, speaker, source_name="transcribed", **kwargs):
    return dict(speaker=speaker, split="train", source=source_name,
                authorized=True, verified=True, phones="abc", audioSha256=str(i),
                kind="none", scored=True, events=[], non_errors=[], unsupported=[], **kwargs)


def test_admission_excludes_speakers_across_corpora_and_rejects_mistake_scripts_and_duplicate_audio():
    rows = [provenance(1, "a"), provenance(2, "b", "help-none"),
            provenance(3, "held", "help-none"),
            dict(provenance(4, "held"), split="test"),
            dict(provenance(5, "mistake", "help-none"), kind="added_word", events=[{}])]
    assert [r["speaker"] for r in admit_recordings(rows, [])] == ["a", "b"]
    with pytest.raises(ValueError, match="Duplicate"):
        admit_recordings(rows + [provenance(1, "other")], [])
    with pytest.raises(ValueError, match="performed transcript"):
        admit_recordings(rows + [dict(provenance(6, "other"), verified=False)], [])


def test_partition_is_speaker_grouped_and_independent_of_input_order():
    names = [f"speaker-{i}" for i in range(10)]
    assert speaker_partition(names) == speaker_partition(list(reversed(names)) + names)
    assert set(speaker_partition(names).values()) == {"fit", "selection"}


class Schedule:
    def __init__(self):
        self.batches, self.epochs = [], []

    def step_batch(self, i):
        self.batches.append(i)

    def step_epoch(self, i):
        self.epochs.append(i)


def recordings(native):
    rows = []
    for i in range(10):
        for origin in ("transcribed", "help-none"):
            x = torch.randn(9, 7, dtype=torch.float64, requires_grad=True)
            rows.append(dict(speaker=f"speaker-{i}", source=origin, views=[
                dict(features=x, target=torch.tensor([1, 2, 3]), shipped=native(x).detach())]))
    return rows


def test_matched_refit_has_equal_gradient_evaluations_schedules_and_no_encoder_gradients():
    native = source()
    rows, schedules = recordings(native), []

    def factory(opt):
        schedule = Schedule()
        schedules.append(schedule)
        return schedule

    heads, report = matched_refit(native, rows,
        lambda params: torch.optim.Adam([p for _, p in params], lr=.002),
        factory, epochs=2, seed=31)
    assert len(set(report["gradientEvaluations"].values())) == 1
    assert all(s.batches == schedules[0].batches and s.epochs == schedules[0].epochs for s in schedules)
    assert report["selectedUpdates"] == report["selectedEpoch"] * 16
    assert all(v["features"].grad is None for r in rows for v in r["views"])
    assert all(p.grad is None for p in native.parameters())
    assert all(not h.training for h in heads.values())


def test_refit_rejects_a_head_that_does_not_reproduce_shipped_probabilities():
    native = source()
    rows = recordings(native)
    rows[0]["views"][0]["shipped"] = torch.zeros_like(rows[0]["views"][0]["shipped"])
    with pytest.raises(AssertionError):
        matched_refit(native, rows, lambda p: None, lambda o: None, epochs=1, seed=31)


def test_real_pause_frames_remain_in_the_preservation_loss_and_invalid_ctc_fails():
    head = source()
    x = torch.zeros(9, 7, dtype=torch.float64)
    shipped = head(x).detach().log_softmax(-1)
    view = dict(features=x, target=torch.tensor([1, 2]), shipped=shipped)
    head.eval()
    kwargs = dict(dropout=.1, ctc_weight=1., kd_weight=0., preserve_weight=1.)
    before = recording_loss(head, [view], **kwargs)
    altered = copy.deepcopy(view)
    altered["shipped"][-3:] = torch.tensor([8., 0., 0., 0., 0.])
    assert recording_loss(head, [altered], **kwargs) != before
    view["target"] = torch.ones(7, dtype=torch.int64)
    with pytest.raises(ValueError, match="transcript"):
        recording_loss(head, [view], **kwargs)
