"""Synthetic enrollment, immutable-cache and paired head-refit contracts."""
import json
import ast
from contextlib import nullcontext
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import student_censored_train as runner
from student_head_refit import ROOT, digest, has_clean_transcript, reserved_speakers, write_json


def test_recorded_sparse_targets_preserve_ratios_and_clamped_frame_clock():
    ids = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype='uint8')
    p = np.array([[.4, .2, .1, .05, .025], [.5, .25, .125, .0625, .03125]], dtype='float16')
    observed_ids, q = runner.sparse_topk(ids.tobytes(), p.tobytes(), 6, 4)
    np.testing.assert_array_equal(observed_ids, ids[[0, 1, 1, 1]])
    np.testing.assert_allclose(q.sum(-1), 1., atol=1e-6)
    np.testing.assert_allclose(q[0], p[0].astype('float32')/p[0].sum(dtype='float32'))
    with pytest.raises(ValueError):
        runner.sparse_topk(np.zeros_like(ids).tobytes(), p.tobytes(), 6, 4)


def test_clean_canonical_supervision_excludes_scripted_errors_and_repeats():
    assert has_clean_transcript({'kind': 'none', 'split': 'train'})
    for change in [{'kind': 'added_word'}, {'events': [{}]}, {'non_errors': [{}]},
                   {'unsupported': ['x']}, {'scored': False}]:
        assert not has_clean_transcript({'kind': 'none', **change})
    rows = [{'speaker': str(i)} for i in range(10)]
    assert reserved_speakers(rows) == reserved_speakers(list(reversed(rows)))
    assert len(reserved_speakers(rows)) == 2


def test_pause_mask_uses_real_samples_without_padded_or_teacher_blank_labels():
    wave = np.ones(6400, dtype='float32')
    wave[1600:4800] = 0
    mask = runner.digital_pause(wave, 14)
    assert mask[4:6].all()
    assert not mask[:3].any() and not mask[9:].any()


def test_selection_uses_censored_training_validation_and_pairs_same_epoch():
    history = [dict(epoch=0, validation={'censored': 3., 'renormalized': 1.}),
               dict(epoch=1, validation={'censored': 2., 'renormalized': 5.}),
               dict(epoch=2, validation={'censored': 2., 'renormalized': .1})]
    assert runner.choose_checkpoint(history)['epoch'] == 1
    history[2]['validation']['censored'] = float('nan')
    with pytest.raises(ValueError):
        runner.choose_checkpoint(history)


def synthetic_plan(out, recipe):
    rng = np.random.default_rng(5)
    weight = rng.normal(scale=.02, size=(6, 3)).astype('float32')
    np.savez(out/'initial-head.npz', weight=weight, bias=np.zeros(6, dtype='float32'))
    (out/'encoder-features.onnx').write_bytes(b'frozen synthetic encoder provenance')
    views = []
    for i, partition in enumerate(['train', 'train', 'validation']):
        x = rng.normal(size=(12, 3)).astype('float32')
        logits = x@weight.T
        lp = logits-np.log(np.exp(logits).sum(-1, keepdims=True))
        path = out/f'view-{i}.npz'
        np.savez(path, features=x, baseline=lp, target=np.array([1, 2], dtype='int64'),
                 pause=np.zeros(12, dtype=bool), top_ids=np.tile(np.arange(5), (12, 1)),
                 top_p=np.tile(np.array([.5, .2, .15, .1, .05], dtype='float32'), (12, 1)))
        views.append(dict(recording=str(i), speaker=str(i), partition=partition, view=0,
                          file=str(path), sha256=digest(path)))
    write_json(out/'plan.json', dict(recipe=recipe, views=views, excluded_speakers=['excluded'],
               quantized=False, initial_head_sha256=digest(out/'initial-head.npz'),
               feature_graph_sha256=digest(out/'encoder-features.onnx')))
    return views


def test_real_fit_keeps_encoder_frozen_and_runs_identical_data_and_update_counts(tmp_path):
    recipe = dict(runner.RECIPE, epochs=2, batch=1)
    views = synthetic_plan(tmp_path, recipe)
    before = {v['file']: digest(v['file']) for v in views}
    with patch.object(runner, 'RECIPE', recipe):
        runner.fit(tmp_path)
    selected = json.loads((tmp_path/'selection.json').read_text())
    assert selected['identical_updates'] and selected['identical_batches']
    assert [h['updates'] for h in selected['history']] == [0, 2, 4]
    for arm, ck in selected['arms'].items():
        assert Path(ck['path']).name == f"{arm}-{selected['selected']['epoch']}.npz"
        assert digest(ck['path']) == ck['sha256']
    assert {v['file']: digest(v['file']) for v in views} == before
    a, b = [np.load(tmp_path/f'{arm}-2.npz')['weight'] for arm in ('renormalized', 'censored')]
    assert not np.array_equal(a, b)


def test_hash_and_speaker_audits_fail_before_refitting(tmp_path):
    views = synthetic_plan(tmp_path, runner.RECIPE)
    plan = json.loads((tmp_path/'plan.json').read_text())
    plan['views'][-1]['speaker'] = plan['views'][0]['speaker']
    write_json(tmp_path/'plan.json', plan)
    with pytest.raises(ValueError, match='separation'):
        runner.read_plan(tmp_path)
    plan['views'][-1]['speaker'] = 'held-training'
    write_json(tmp_path/'plan.json', plan)
    Path(views[0]['file']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Frozen features'):
        runner.read_plan(tmp_path)


def test_trainer_routes_censored_kd_and_preserves_existing_default(monkeypatch):
    import torch
    tree = ast.parse((ROOT/'modal/student_train.py').read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'losses']
    scope = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'student_train.py', 'exec'), scope)
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    probabilities = torch.tensor([.4, .2, .16, .12, .08, .04])
    lp = probabilities.log()[None, None].repeat(1, 3, 1)
    model = lambda feats, lengths: (lp, torch.tensor([3]))
    batch = dict(feats=torch.zeros(1, 3, 80), feat_lens=torch.tensor([3]),
                 targets=torch.tensor([1]), target_lens=torch.tensor([1]), n_teacher=torch.tensor([3]),
                 top_ids=torch.arange(5)[None, None].repeat(1, 3, 1),
                 top_p=probabilities[:5][None, None].repeat(1, 3, 1))
    cfg = dict(ctc_weight=1., kd_weight=1., kd_delay=0)
    old = scope['losses'](model, batch, cfg, 'cpu')
    explicit = scope['losses'](model, batch, dict(cfg, kd_target='renormalized'), 'cpu')
    censored = scope['losses'](model, batch, dict(cfg, kd_target='censored'), 'cpu')
    assert old['kd'] == explicit['kd']
    assert abs(float(censored['kd'])) < 1e-6
    assert old['ctc'] == censored['ctc']
    with pytest.raises(ValueError, match='kd_target'):
        scope['losses'](model, batch, dict(cfg, kd_target='unknown'), 'cpu')
