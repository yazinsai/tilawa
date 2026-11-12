import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student_temporal import (admitted_none, speaker_partition, ctc_occupancy,
    occurrence_intervals, consensus_pairs, stable_pairs, temporal_loss,
    phone_head, log_probs, objectives, recording_loss, PAIR_RULE, transcript_target)
from student_temporal_export import graft, expose, session, stream, digest

def example():
    x = torch.tensor([[1., .1], [.5, .2], [.2, .7], [.1, 1.], [.2, .4], [.1, .1]])
    shipped = torch.tensor([[3., 0., 0.]]).repeat(len(x), 1).log_softmax(-1)
    return dict(features=x, shipped=shipped, teacher=shipped.clone(), target=torch.tensor([1, 2]))


def test_transcript_duration_notation_uses_shipped_ctc_run_collapse():
    assert transcript_target("abbccab") == "abcab"


@pytest.mark.parametrize("target", [[1], [1, 2], [1, 1]])
def test_ctc_marginals_match_exhaustive_collapsed_paths(target):
    from student import ctc_collapse
    p = torch.tensor([[.4, .5, .1], [.5, .2, .3], [.2, .5, .3], [.5, .4, .1]], dtype=torch.float64)
    expected, blank, total = torch.zeros(4, len(target), dtype=torch.float64), torch.zeros(4, dtype=torch.float64), 0.
    for path in itertools.product(range(3), repeat=4):
        if ctc_collapse(path) != target:
            continue
        weight = float(torch.stack([p[t, phone] for t, phone in enumerate(path)]).prod())
        total += weight
        occurrence, prev = -1, -1
        for t, phone in enumerate(path):
            if phone == 0:
                blank[t] += weight
            else:
                if phone != prev:
                    occurrence += 1
                expected[t, occurrence] += weight
            prev = phone
    got, got_blank, log_z = ctc_occupancy(p.log(), torch.tensor(target))
    torch.testing.assert_close(got, expected / total)
    torch.testing.assert_close(got_blank, blank / total)
    assert float(log_z.exp()) == pytest.approx(total)
    assert not got.requires_grad


def test_repeated_phone_occurrences_have_distinct_neighbourhoods():
    lp = torch.full((9, 3), -8.)
    lp[:, 0] = 0.
    lp[2, 1] = lp[6, 1] = 8.
    intervals = occurrence_intervals(lp, torch.tensor([1, 1]))
    assert intervals == [(0, 7), (2, 9)]
    # Align the full teacher, but no comparison can reference its trailing rows.
    pairs = stable_pairs(lp, lp.clone(), torch.tensor([1, 1]), 5)
    assert pairs and all(int(p[:, 1:].max()) < 5 for p in pairs)


def test_only_strict_agreed_ordering_and_no_magnitude_targets():
    a = torch.zeros(4, 3)
    b = torch.zeros(4, 3)
    a[:, 1] = torch.tensor([-4., -1., -2., -2.])
    b[:, 1] = torch.tensor([-5., -3., -1., -1.])
    pairs = consensus_pairs(a, b, 1, 0, 4, 0)
    assert set(map(tuple, pairs.tolist())) == {(1, 1, 0), (1, 2, 0), (1, 3, 0)}
    transformed = consensus_pairs(a * 20 - 8, b * .1 + 2, 1, 0, 4, 0)
    assert torch.equal(pairs, transformed)
    assert not consensus_pairs(a, -a, 1, 0, 4, 0).numel()


def test_comparison_budget_and_hash_sampling_are_deterministic():
    lp = torch.zeros(40, 3)
    lp[:, 1] = torch.arange(40)
    first = consensus_pairs(lp, lp, 1, 0, 40, 3)
    assert len(first) == PAIR_RULE["maxPairsPerOccurrence"]
    assert torch.equal(first, consensus_pairs(lp * 2, lp * 3, 1, 0, 40, 3))
    assert all(i > j for _, i, j in first.tolist())
    assert len(set(map(tuple, first.tolist()))) == len(first)


def test_rank_loss_uses_equal_occurrences_and_pushes_only_order():
    lp = torch.tensor([[-1., -3.], [-1., -2.], [-1., -4.]], requires_grad=True)
    one = torch.tensor([[1, 1, 0]])
    two = torch.tensor([[1, 1, 2], [1, 0, 2]])
    got = temporal_loss(lp, [one, two])
    expected = (torch.nn.functional.softplus(-torch.tensor(1.)) +
                torch.nn.functional.softplus(-torch.tensor([2., 1.])).mean()) / 2
    torch.testing.assert_close(got, expected)
    assert torch.equal(got, temporal_loss(lp, [one.repeat(7, 1), two]))
    got.backward()
    assert lp.grad[1, 1] < 0 and lp.grad[2, 1] > 0
    assert torch.equal(lp.grad[:, 0], torch.zeros(3))
    assert temporal_loss(lp, []) == 0


@pytest.mark.parametrize("arm", ["ctc", "ordinary", "rank"])
def test_matched_arms_equal_their_declared_objective_and_detach_evidence(arm):
    head, v = phone_head(2, 3), example()
    views = [copy.deepcopy(v) for _ in range(3)]
    for w in views:
        for key in ("features", "shipped", "teacher"):
            w[key].requires_grad_(True)
    pairs = [torch.tensor([[1, 0, 1], [1, 1, 2]])]
    got = recording_loss(head, views, pairs, arm)
    predicted = [log_probs(head, w["features"], w["shipped"]) for w in views]
    components = [objectives(p, w["target"], w["teacher"], w["shipped"]) for p, w in zip(predicted, views)]
    expected = sum(c + preserve + (kd if arm == "ordinary" else 0) for c, kd, preserve in components) / 3
    if arm == "rank":
        expected = expected + sum(temporal_loss(p, pairs) for p in predicted[1:]) / 2
    g1 = torch.autograd.grad(got, tuple(head.parameters()), retain_graph=True)
    g2 = torch.autograd.grad(expected, tuple(head.parameters()))
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b)
    assert all(w[k].grad is None for w in views for k in ("features", "shipped", "teacher"))


def test_complete_student_ctc_and_pause_preservation_do_not_depend_on_teacher_edges():
    v = example()
    lp = v["shipped"]
    full = objectives(lp, v["target"], v["teacher"], lp)
    for t in (v["teacher"][:-1], torch.cat([v["teacher"], v["teacher"][:1]])):
        short = objectives(lp, v["target"], t, lp)
        assert torch.equal(full[0], short[0]) and torch.equal(full[2], short[2])


def test_invalid_alignments_pairs_and_arms_fail():
    with pytest.raises(ValueError, match="Impossible"):
        ctc_occupancy(torch.zeros(2, 3), torch.tensor([1, 1]))
    with pytest.raises(ValueError):
        stable_pairs(torch.full((4, 3), float("nan")), torch.zeros(4, 3), torch.tensor([1]), 4)
    for p in (torch.tensor([[0, 1, 0]]), torch.tensor([[1, 0, 0]]), torch.tensor([[1, 0, 4]])):
        with pytest.raises(ValueError):
            temporal_loss(torch.zeros(4, 3), [p])
    with pytest.raises(ValueError):
        recording_loss(phone_head(2, 3), [example()] * 3, [], "unknown")


def none_row(i, speaker):
    return dict(id=str(i), speaker=speaker, split="train", kind="none", scored=True,
                events=[], non_errors=[], unsupported=[], wav="take.wav", ayahs=["1:1"])


def test_only_explicit_none_from_consistently_training_speakers():
    clean = none_row(1, "a")
    repeat = {**none_row(2, "b"), "kind": "repeat"}
    hidden = {**none_row(3, "c"), "unsupported": [{"kind": "vowel"}]}
    held = {**none_row(4, "a"), "split": "test"}
    assert admitted_none([clean, repeat, hidden]) == [clean]
    assert admitted_none([clean, held]) == []
    assert admitted_none([clean], ["a"]) == []
    assert admitted_none([{**clean, "scored": False}]) == []
    assert admitted_none([{k: v for k, v in clean.items() if k != "non_errors"}]) == []
    with pytest.raises(ValueError, match="Duplicate"):
        admitted_none([clean, clean])


def test_speaker_selection_is_stable_and_groups_all_takes():
    speakers = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    p = speaker_partition(speakers)
    assert p == speaker_partition(list(reversed(speakers)) + ["a", "b"])
    assert set(p.values()) == {"fit", "selection"}
    assert sum(v == "selection" for v in p.values()) == 2
    with pytest.raises(ValueError):
        speaker_partition(["a", "b"])


def enrollment(tmp_path, extra_updates=None):
    from student_temporal_train import recordings
    from student import TOKENS
    help_audio, extra_audio = tmp_path / "help.wav", tmp_path / "extra.wav"
    help_audio.write_bytes(b"first enrolled recording")
    extra_audio.write_bytes(b"different enrolled recording")
    help_labels, extra_labels = tmp_path / "help.jsonl", tmp_path / "transcribed.jsonl"
    help_row = {**none_row(1, "help-speaker"), "wav": help_audio.name}
    extra_row = dict(speaker="extra-speaker", split="train", authorized=True,
                     verified=True, phones=TOKENS[0], wav=extra_audio.name,
                     sha256=digest(extra_audio))
    extra_row.update(extra_updates or {})
    help_labels.write_text(json.dumps(help_row) + "\n")
    extra_labels.write_text(json.dumps(extra_row) + "\n")
    reference = dict(ayahs=[dict(s=1, a=1, p=TOKENS[:2])])
    return recordings, help_labels, extra_labels, reference


def test_cross_corpus_speaker_exclusion_precedes_enrollment(tmp_path):
    enroll, help_labels, extra_labels, reference = enrollment(tmp_path)
    rows, excluded = enroll(help_labels, extra_labels, [], reference)
    assert {r["source"] for r in rows} == {"help-none", "transcribed"}
    assert excluded == []
    # A non-training entry in either corpus excludes all that speaker's takes,
    # even an otherwise admissible none take in the other corpus.
    with extra_labels.open("a") as out:
        out.write(json.dumps(dict(speaker="help-speaker", split="test")) + "\n")
    with pytest.raises(ValueError, match="Both authorized"):
        enroll(help_labels, extra_labels, [], reference)


@pytest.mark.parametrize("update", [dict(verified=False), dict(authorized=False),
                                   dict(sha256="0" * 64)])
def test_unverified_or_changed_transcribed_recordings_fail(tmp_path, update):
    enroll, help_labels, extra_labels, reference = enrollment(tmp_path, update)
    with pytest.raises(ValueError):
        enroll(help_labels, extra_labels, [], reference)


def test_duplicate_audio_across_corpora_fails(tmp_path):
    enroll, help_labels, extra_labels, reference = enrollment(tmp_path)
    extra = json.loads(extra_labels.read_text())
    extra.update(wav="help.wav", sha256=digest(tmp_path / "help.wav"))
    extra_labels.write_text(json.dumps(extra) + "\n")
    with pytest.raises(ValueError, match="Repeated audio"):
        enroll(help_labels, extra_labels, [], reference)


def test_selection_weights_speakers_equally_and_never_updates_head():
    from student_temporal_train import validate
    head, first = phone_head(2, 3), example()
    second = {**first, "target": torch.tensor([2, 1, 2])}
    rows = [dict(speaker="one", views=[first]), dict(speaker="two", views=[second])]
    before = copy.deepcopy(head.state_dict())
    metrics = validate(head, rows)
    duplicated = validate(head, [rows[0]] * 9 + [rows[1]])
    assert metrics == duplicated and metrics["speakers"] == 2
    assert all(torch.equal(value, head.state_dict()[key]) for key, value in before.items())
    assert all(p.grad is None for p in head.parameters())
    with pytest.raises(ValueError, match="selection"):
        validate(head, [])


def source_bundle(path, tokens="ab"):
    import onnx
    from onnx import helper as h, TensorProto as T, numpy_helper as n
    path.mkdir()
    # Stateful native feature emitter, with the same frame geometry as SDK.
    nodes = [h.make_node("ReduceMean", ["x"], ["mean"], axes=[1], keepdims=1),
             h.make_node("Expand", ["mean", "shape"], ["feature"]),
             h.make_node("MatMul", ["feature", "weight"], ["scores"], name="ctc_output/MatMul"),
             h.make_node("LogSoftmax", ["scores"], ["log_probs"], axis=-1),
             h.make_node("Add", ["state", "one"], ["new_state"])]
    g = h.make_graph(nodes, "test", [h.make_tensor_value_info("x", T.FLOAT, [1, 45, 80]),
        h.make_tensor_value_info("state", T.INT64, [1])],
        [h.make_tensor_value_info("log_probs", T.FLOAT, [1, 8, len(tokens) + 1]), h.make_tensor_value_info("new_state", T.INT64, [1])],
        [n.from_array(np.array([1, 8, 80], dtype=np.int64), "shape"),
         n.from_array(np.random.default_rng(0).normal(0, .01, (80, len(tokens) + 1)).astype(np.float32), "weight"),
         n.from_array(np.ones(1, dtype=np.int64), "one")])
    model = h.make_model(g, opset_imports=[h.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, path / "model.onnx")
    # Real external tensor storage exercises complete auxiliary bundle copying.
    teacher = h.make_model(h.make_graph([], "teacher", [], [],
        [n.from_array(np.ones(4, dtype=np.float32), "teacher_weight")]), ir_version=9)
    onnx.save_model(teacher, path / "teacher.onnx", save_as_external_data=True,
                   all_tensors_to_one_file=True, location="teacher.onnx.data", size_threshold=0)
    manifest = dict(file="model.onnx", sha256=digest(path / "model.onnx"), id="unit", tokens=tokens, blank=0,
                    frameMs=40, sampleRate=16000, window=45, shift=32, fbank=dict(mels=80, padValue=-23.0259),
                    states=[dict(name="state", shape=[1], dtype="int64")])
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_export_preserves_caches_clock_assets_and_selected_head(tmp_path):
    source = tmp_path / "source"
    manifest = source_bundle(source)
    m, feature, width, extractor = expose(source, tmp_path / "extraction")
    assert width == 80
    head = phone_head(width, 3)
    with torch.no_grad():
        head.bias.copy_(torch.tensor([.2, -.1, .3]))
        head.weight[:, :2] = torch.tensor([[.1, .2], [.3, -.1], [0., .2]])
    new = graft(source, head, tmp_path / "export", "rank")
    new_m = json.loads((new.parent / "manifest.json").read_text())
    f = np.random.default_rng(1).normal(-10, 2, (99, 80)).astype(np.float32)
    old, x = stream(session(extractor), m, f, feature)
    got, _ = stream(session(new), new_m, f)
    with torch.no_grad():
        want = log_probs(head, torch.tensor(x), torch.tensor(old)).numpy()
    np.testing.assert_allclose(got, want, atol=2e-6)
    for key in ("states", "window", "shift", "fbank", "tokens", "blank", "frameMs", "sampleRate"):
        assert manifest[key] == new_m[key]
    for name in ("teacher.onnx", "teacher.onnx.data"):
        assert digest(source / name) == digest(new.parent / name)
    feeds = dict(x=f[:45][None], state=np.array([3], dtype=np.int64))
    np.testing.assert_array_equal(session(source / "model.onnx").run(["new_state"], feeds)[0],
                                  session(new).run(["new_state"], feeds)[0])
    assert digest(source / "model.onnx") == manifest["sha256"]
    assert digest(new) == new_m["sha256"]
    with pytest.raises(ValueError, match="empty"):
        graft(source, head, tmp_path / "export", "rank")
    (source / "teacher.onnx.data").unlink()
    with pytest.raises(ValueError, match="missing"):
        graft(source, head, tmp_path / "bad-export", "rank")


def test_zero_head_export_preserves_distribution(tmp_path):
    source = tmp_path / "source"
    m = source_bundle(source)
    path = graft(source, phone_head(80, 3), tmp_path / "export", "ordinary")
    features = np.zeros((45, 80), dtype=np.float32)
    old, _ = stream(session(source / "model.onnx"), m, features)
    got, _ = stream(session(path), m, features)
    np.testing.assert_allclose(old, got, atol=1e-7)


def test_int8_encoder_extraction_and_optimized_export_preserve_source(tmp_path):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    source = tmp_path / "source"
    manifest = source_bundle(source)
    quantized = source / "int8.onnx"
    quantize_dynamic(source / "model.onnx", quantized, weight_type=QuantType.QInt8)
    quantized.replace(source / "model.onnx")
    manifest["sha256"] = digest(source / "model.onnx")
    (source / "manifest.json").write_text(json.dumps(manifest))
    old_graph = onnx.load(source / "model.onnx")
    assert any(n.op_type == "MatMulInteger" for n in old_graph.graph.node)
    m, feature, width, extractor = expose(source, tmp_path / "extraction")
    head = phone_head(width, 3)
    with torch.no_grad():
        head.weight[:, :2] = torch.tensor([[.1, .2], [.3, -.1], [0., .2]])
    path = graft(source, head, tmp_path / "export", "rank")
    features = np.random.default_rng(3).normal(-10., 2., (99, 80)).astype(np.float32)
    original, _ = stream(session(source / "model.onnx"), m, features)
    lp, x = stream(session(extractor), m, features, feature)
    np.testing.assert_array_equal(original, lp)
    with torch.no_grad():
        expected = log_probs(head, torch.tensor(x), torch.tensor(lp)).numpy()
    got, _ = stream(session(path, optimize=True), m, features)
    np.testing.assert_allclose(np.exp(got), np.exp(expected), atol=2e-6, rtol=2e-6)
    new_weights = {t.name: t.SerializeToString() for t in onnx.load(path).graph.initializer}
    assert all(new_weights[t.name] == t.SerializeToString() for t in old_graph.graph.initializer)


def test_shared_cache_three_arm_fit_and_complete_exports(tmp_path, monkeypatch):
    import argparse
    import student_temporal_train as train
    from student import TOKENS
    from student_temporal_export import insertion_assets
    source, cache, out = tmp_path / "source", tmp_path / "cache", tmp_path / "fit"
    manifest = source_bundle(source, TOKENS)
    m, feature, width, extracted = expose(source, cache)
    speakers = ["a", "b", "c", "d", "e"]
    groups, rows = speaker_partition(speakers), []
    for i, speaker in enumerate(speakers):
        views = []
        for view in range(3):
            f = np.random.default_rng(100 + i * 3 + view).normal(-10, 2, (45, 80)).astype(np.float32)
            shipped, x = stream(session(extracted), manifest, f, feature)
            teacher = np.full(shipped.shape, -8., dtype=np.float32)
            teacher[:, 0] = 0.
            teacher[2, 1], teacher[5, 2] = 6., 6.
            file = f"r{i}-v{view}.npz"
            np.savez(cache / file, features=x, shipped=shipped, teacher=teacher,
                     target=np.array([1, 2], dtype=np.int64), fbank=f)
            views.append(file)
        rows.append(dict(speaker=speaker, views=views,
                         viewHashes=[digest(cache / file) for file in views]))
    info = dict(source=str(source), sourceSha256=m["sha256"], width=width,
                tokens=TOKENS, groups=groups, rows=rows, excludedSpeakers=[],
                auxiliaryHashes=insertion_assets(source), pairRule=PAIR_RULE,
                preparationSeconds=0.)
    train.write_json(cache / "prepared.json", info)
    monkeypatch.setattr(train, "EPOCHS", 2)
    train.fit(argparse.Namespace(prepared=str(cache), out=str(out)))
    report = json.loads((out / "training-report.json").read_text())
    assert set(report["arms"]) == {"ctc", "ordinary", "rank"}
    assert report["comparisons"] > 0
    for arm, result in report["arms"].items():
        assert len(result["history"]) == 3 and 0 <= result["selectedEpoch"] <= 2
        assert result["parityRecordings"] == 1
        assert digest(out / arm / "model.onnx") == result["sha256"]
        assert insertion_assets(out / arm) == insertion_assets(source)
    # A changed pairing rule invalidates this cache before any head updates.
    info["pairRule"] = {**PAIR_RULE, "version": 2}
    train.write_json(cache / "prepared.json", info)
    with pytest.raises(ValueError, match="Pairing rule"):
        train.load_prepared(cache)


def test_shared_training_budget_is_enforced(monkeypatch):
    import student_temporal_train as train
    monkeypatch.setattr(train.time, "monotonic", lambda: 10.)
    train.check_budget(0., 0.)
    with pytest.raises(RuntimeError, match="one-day"):
        train.check_budget(train.BUDGET_SECONDS - 10., 0.)
