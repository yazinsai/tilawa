"""pytest modal/test_student_noise.py (numpy, torch); no models or audio needed."""

import itertools
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_noise import (Observation, channel_posterior, corrected_distillation,
                           ctc_occupancy, fit_channel, head_loss, speaker_partition)
from student_head_refit import digest, has_clean_transcript, write_json
from student_noise_train import (RECIPE, choose_checkpoint, digital_pause, fit,
                                 full_topk, quantized_parameters, read_plan)


def collapse(path):
    return tuple(p for i, p in enumerate(path) if p and (i == 0 or p != path[i - 1]))


@pytest.mark.parametrize("target", [(), (1,), (1, 1), (1, 2)])
def test_forward_backward_matches_exhaustive_ctc_paths(target):
    p = np.array([[.6, .3, .1], [.2, .5, .3], [.4, .1, .5], [.7, .2, .1]])
    pause = np.array([False, False, True, False])
    paths = [s for s in itertools.product(range(3), repeat=4)
             if collapse(s) == target and s[2] == 0]
    weights = np.array([np.prod([p[t, s[t]] for t in range(4)]) for s in paths])
    expected = np.zeros_like(p)
    for s, weight in zip(paths, weights):
        for t, phone in enumerate(s):
            expected[t, phone] += weight / weights.sum()
    log_z, posterior = ctc_occupancy(np.log(p), target, pause)
    assert np.exp(log_z) == pytest.approx(weights.sum())
    np.testing.assert_allclose(posterior, expected, atol=1e-12)
    np.testing.assert_allclose(posterior.sum(-1), 1)


def test_impossible_repeated_phone_and_pause_constraints_are_rejected():
    with pytest.raises(ValueError, match="no legal path"):
        ctc_occupancy(np.zeros((2, 3)), (1, 1))
    with pytest.raises(ValueError, match="no legal path"):
        ctc_occupancy(np.zeros((3, 3)), (1,), np.ones(3, dtype=bool))


def test_channel_learns_blanking_without_using_teacher_argmax_as_latent_truth():
    # Every recording must contain phone 1, although the teacher observes only
    # blank. A forced argmax label estimator cannot recover this confusion row.
    q = np.tile([.99, .005, .005], (8, 1))
    pause = np.array([True, True, False, False, False, False, True, True])
    items = [Observation(f"speaker-{i}", (1,), q, pause) for i in range(20)]
    channel = fit_channel(items, diagonal_frames=1, iterations=6)
    np.testing.assert_allclose(channel.sum(-1), 1)
    assert channel[1, 0] > .85
    assert channel[2, 2] > .99  # unseen latent class preserved by the prior
    assert not channel.flags.writeable
    _, occupancy = channel_posterior(channel, items[0])
    assert occupancy[:, 1].sum() >= 1
    assert not occupancy[pause, 1:].any()


def test_identity_is_ordinary_kd_and_channel_is_frozen():
    logits = torch.tensor([[[2., 1., -1.], [1., 0., 3.]]], requires_grad=True)
    q = torch.tensor([[[.1, .7, .2], [.3, .2, .5]]])
    valid = torch.tensor([[True, False]])
    channel = torch.eye(3, requires_grad=True)
    lp = logits.log_softmax(-1)
    loss = corrected_distillation(lp, q, valid, channel)
    expected = -(lp[0, 0] * q[0, 0]).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert channel.grad is None
    assert logits.grad[0, 1].abs().sum() == 0


def test_forward_correction_allows_phone_to_explain_teacher_blank():
    channel = torch.tensor([[.99, .01], [.8, .2]])
    logits = torch.tensor([[[.05, .95]]]).log().requires_grad_()
    lp = logits.log_softmax(-1)
    q = torch.tensor([[[.8, .2]]])
    valid = torch.ones((1, 1), dtype=torch.bool)
    corrected = corrected_distillation(lp, q, valid, channel)
    direct = corrected_distillation(lp, q, valid, torch.eye(2))
    assert corrected < direct
    corrected_gradient = torch.autograd.grad(corrected, logits, retain_graph=True)[0]
    direct_gradient = torch.autograd.grad(direct, logits)[0]
    assert corrected_gradient[0, 0, 1] < 0  # correction asks for more latent phone
    assert direct_gradient[0, 0, 1] > 0  # direct KD suppresses that same phone
    with pytest.raises(ValueError, match="normalized"):
        corrected_distillation(lp, q, valid, channel * 2)


def test_pause_supervises_latent_blank_and_padding_never_contributes():
    logits = torch.tensor([[[2., 0.], [0., 2.], [1., 1.], [-9., 9.]]], requires_grad=True)
    lp = logits.log_softmax(-1)
    q = torch.ones_like(lp) / 2
    lengths, target_lengths = torch.tensor([3]), torch.tensor([1])
    pause = torch.tensor([[True, False, False, True]])
    loss, parts = head_loss(lp, q, torch.tensor([1]), lengths, target_lengths, pause,
                           torch.tensor([[.5, .5], [.9, .1]]))
    torch.testing.assert_close(parts["pause"], -lp[0, 0, 0])
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 3].abs().sum() == 0


def test_group_assignment_keeps_a_speakers_takes_together():
    assignments = {s: speaker_partition(s) for s in ("speaker-a", "speaker-b", "speaker-c")}
    for s in assignments:
        assert speaker_partition(s) == assignments[s]
    with pytest.raises(ValueError):
        speaker_partition("")


def test_scripted_mistakes_and_repeats_never_become_canonical_transcripts():
    assert has_clean_transcript({'kind': 'none', 'split': 'train'})
    for label in ({'kind': 'wrong_word'}, {'kind': 'none', 'non_errors': [{'kind': 'repeat'}]},
                  {'kind': 'none', 'events': [{'kind': 'omitted_word'}]}, {'kind': 'none', 'scored': False}):
        assert not has_clean_transcript(label)


def test_silence_mask_uses_real_samples_never_padding_or_teacher_blank():
    wave = np.zeros(6400, dtype=np.float32)
    wave[3200] = .1
    mask = digital_pause(wave, 15)
    assert mask[2]
    assert not mask[4]  # a single real non-silent sample excludes the neighborhood
    assert not mask[0]  # neighborhood crosses recording start
    assert not mask[-1]  # neighborhood crosses recording end


def test_topk_targets_keep_soft_mass_and_existing_frame_mapping():
    ids = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=np.uint8)
    p = np.array([[.4, .2, .1, .05, .05], [.5, .1, .1, .1, .1]], dtype=np.float16)
    q = full_topk(ids.tobytes(), p.tobytes(), 5, 4)
    np.testing.assert_allclose(q.sum(-1), 1)
    assert 0 < q[0, 1] < q[0, 0] < 1
    np.testing.assert_array_equal(q[1], q[2])
    np.testing.assert_array_equal(q[2], q[3])


def test_checkpoint_uses_corrected_training_validation_and_pairs_same_update():
    history = [dict(epoch=0, updates=0, validation={'corrected': 3., 'ordinary': 1.}),
               dict(epoch=1, updates=4, validation={'corrected': 2., 'ordinary': 3.}),
               dict(epoch=2, updates=8, validation={'corrected': 2., 'ordinary': .5})]
    assert choose_checkpoint(history) == history[1]


def test_selection_uses_exported_weight_rounding():
    head = torch.nn.Linear(3, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1., .013, .67], [.1, .015, .053]]))
    for per_channel in (False, True):
        weight, _ = quantized_parameters(head, dict(quantized=True, per_channel=per_channel))
        original = head.weight.detach().numpy()
        scale = np.abs(original).max(1, keepdims=True) / 127 if per_channel else np.abs(original).max() / 127
        np.testing.assert_allclose(weight.numpy(), np.clip(np.round(original / scale), -127, 127) * scale)


def synthetic_plan(root, recipe):
    root.mkdir(exist_ok=True)
    initial = {'weight': np.array([[1., 0.], [-1., 0.]], dtype=np.float32),
               'bias': np.zeros(2, dtype=np.float32)}
    np.savez(root / 'initial-head.npz', **initial)
    (root / 'encoder-features.onnx').write_bytes(b'frozen synthetic encoder')
    views = []
    for i, (speaker, partition) in enumerate([('train-a', 'train'), ('train-b', 'train'), ('validation-a', 'validation')]):
        x = np.tile(np.array([1., .2], dtype=np.float32), (8, 1))
        lp = torch.tensor(x @ initial['weight'].T).log_softmax(-1).numpy()
        file = root / f'features-{i}.npz'
        np.savez(file, features=x, baseline=lp, target=np.array([1], dtype=np.int64),
                 teacher=np.tile(np.array([.9, .1], dtype=np.float32), (8, 1)),
                 pause=np.array([True, True, False, False, False, False, True, True]))
        views.append(dict(file=str(file), sha256=digest(file), view=0, speaker=speaker, partition=partition))
    plan = dict(recipe=recipe, views=views, excluded_speakers=[], quantized=False, per_channel=False,
                initial_head_sha256=digest(root / 'initial-head.npz'),
                feature_graph_sha256=digest(root / 'encoder-features.onnx'))
    write_json(root / 'plan.json', plan)
    return plan


def test_refits_share_updates_freeze_channel_and_select_only_training_speakers(tmp_path):
    recipe = dict(RECIPE, epochs=2, channel_iterations=2, batch=1)
    plan = synthetic_plan(tmp_path, recipe)
    with patch('student_noise_train.RECIPE', recipe):
        fit(tmp_path)
        read_plan(tmp_path)
    selection = json.loads((tmp_path / 'selection.json').read_text())
    assert selection['identical_updates'] and selection['identical_batches']
    assert [h['updates'] for h in selection['history']] == [0, 2, 4]
    assert selection['channel_sha256'] == digest(tmp_path / 'channel.npy')
    assert set(selection['arms']) == {'ordinary', 'corrected'}
    for checkpoint in selection['arms'].values():
        assert digest(checkpoint['path']) == checkpoint['sha256']
    # Any change to a cached observation makes continuation invalid.
    Path(plan['views'][0]['file']).write_bytes(b'changed cache')
    with patch('student_noise_train.RECIPE', recipe), pytest.raises(ValueError, match='changed'):
        read_plan(tmp_path)


def test_plan_rejects_cross_partition_speaker_and_release_exclusion(tmp_path):
    plan = synthetic_plan(tmp_path, RECIPE)
    plan['views'][-1]['speaker'] = plan['views'][0]['speaker']
    write_json(tmp_path / 'plan.json', plan)
    with pytest.raises(ValueError, match='separation'):
        read_plan(tmp_path)
    plan['views'][-1]['speaker'] = 'validation-a'
    plan['excluded_speakers'] = ['train-a']
    write_json(tmp_path / 'plan.json', plan)
    with pytest.raises(ValueError, match='separation'):
        read_plan(tmp_path)
