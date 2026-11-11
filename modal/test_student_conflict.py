import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_conflict import admitted_none, speaker_partition, project_distillation, phone_head, log_probs, objectives, recording_gradient
from student_conflict_export import graft, expose, session, stream, digest


def dot(a, b):
    return sum((x.double() * y.double()).sum() for x, y in zip(a, b))


@pytest.mark.parametrize("kd", [[2., 3.], [0., 3.], [0., 0.]])
def test_agreeing_or_orthogonal_gradients_are_exact(kd):
    ctc, teacher = (torch.tensor([1., 0.]),), (torch.tensor(kd),)
    got, diag = project_distillation(ctc, teacher)
    assert not diag["conflict"]
    assert torch.equal(got[0], teacher[0])
    assert got[0].data_ptr() != teacher[0].data_ptr()


def test_negative_component_removed_orthogonal_component_kept():
    ctc, teacher = (torch.tensor([2., 0.]),), (torch.tensor([-3., 7.]),)
    got, diag = project_distillation(ctc, teacher)
    assert diag["conflict"]
    torch.testing.assert_close(got[0], torch.tensor([0., 7.]))
    assert dot(ctc, got) == 0


def test_projection_covers_weights_and_bias_as_one_vector():
    ctc = (torch.tensor([1.]), torch.tensor([2.]))
    teacher = (torch.tensor([-3.]), torch.tensor([1.]))
    got, _ = project_distillation(ctc, teacher)
    # Global dot=-1, norm=5: individual-tensor clipping would give [0,1].
    torch.testing.assert_close(got[0], torch.tensor([-2.8]))
    torch.testing.assert_close(got[1], torch.tensor([1.4]))
    assert abs(float(dot(ctc, got))) < 1e-6


def test_zero_ctc_and_tiny_nonzero_ctc():
    teacher = (torch.tensor([-2., 4.], dtype=torch.float64),)
    got, diag = project_distillation((torch.zeros(2, dtype=torch.float64),), teacher)
    assert not diag["conflict"]
    assert torch.equal(got[0], teacher[0])
    got, diag = project_distillation((torch.tensor([1e-100, 0.], dtype=torch.float64),), teacher)
    assert diag["conflict"]
    torch.testing.assert_close(got[0], torch.tensor([0., 4.], dtype=torch.float64), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_gradients_fail(bad):
    with pytest.raises(ValueError):
        project_distillation((torch.tensor([bad]),), (torch.tensor([1.]),))
    with pytest.raises(ValueError):
        project_distillation((torch.tensor([1.]),), (torch.tensor([1., 2.]),))


def test_filter_before_recording_average():
    # Opposite transcript directions cancel if averaged first; per-recording
    # filtering still removes the first recording's destructive teacher term.
    a, b = (torch.tensor([1., 0.]),), (torch.tensor([-1., 0.]),)
    kd_a, kd_b = (torch.tensor([-2., 1.]),), (torch.tensor([-2., 1.]),)
    ga, _ = project_distillation(a, kd_a)
    gb, _ = project_distillation(b, kd_b)
    per_record = (ga[0] + gb[0]) / 2
    averaged, _ = project_distillation(((a[0] + b[0]) / 2,), ((kd_a[0] + kd_b[0]) / 2,))
    torch.testing.assert_close(per_record, torch.tensor([-1., 1.]))
    assert not torch.equal(per_record, averaged[0])


def example():
    x = torch.tensor([[1., .1], [.5, .2], [.2, .7], [.1, 1.], [.2, .4], [.1, .1]])
    shipped = torch.tensor([[3., 0., 0.]]).repeat(len(x), 1).log_softmax(-1)
    return dict(features=x, shipped=shipped, teacher=shipped.clone(), target=torch.tensor([1, 2]))


def test_ordinary_matches_autograd_and_filtered_protects_transcript():
    head, v = phone_head(2, 3), example()
    for key in ("features", "shipped", "teacher"):
        v[key].requires_grad_(True)
    ordinary, diag = recording_gradient(head, [v], "ordinary")
    lp = log_probs(head, v["features"], v["shipped"])
    losses = objectives(lp, v["target"], v["teacher"], v["shipped"])
    expected = torch.autograd.grad(sum(losses), tuple(head.parameters()))
    for a, b in zip(ordinary, expected):
        torch.testing.assert_close(a, b)
    # Start away from shipped so blank-heavy imitation opposes transcript.
    with torch.no_grad():
        head.bias[0] = -1
        head.bias[1:] = .5
    filtered, diag = recording_gradient(head, [v], "filtered", preservation_weight=0.)
    ordinary, _ = recording_gradient(head, [v], "ordinary", preservation_weight=0.)
    lp = log_probs(head, v["features"], v["shipped"])
    ctc, kd, _ = objectives(lp, v["target"], v["teacher"], v["shipped"])
    gc = torch.autograd.grad(ctc, tuple(head.parameters()), retain_graph=True)
    gk = torch.autograd.grad(kd, tuple(head.parameters()))
    assert diag["conflict"]
    torch.testing.assert_close(dot(gc, tuple(a - b for a, b in zip(filtered, gc))), torch.tensor(0., dtype=torch.float64), atol=1e-6, rtol=0)
    assert dot(gc, filtered) > dot(gc, ordinary)
    assert dot(gc, gk) < 0
    assert all(v[k].grad is None for k in ("features", "shipped", "teacher"))


def test_views_combined_within_recording_before_projection():
    head, v = phone_head(2, 3), example()
    with torch.no_grad():
        head.bias[0] = -.8
    w = {**v, "features": v["features"].flip(0)}
    got, _ = recording_gradient(head, [v, w], "filtered")
    vals = [objectives(log_probs(head, row["features"], row["shipped"]), row["target"],
                       row["teacher"], row["shipped"]) for row in [v, w]]
    gc, gk, gp = [torch.autograd.grad(sum(r[i] for r in vals) / 2, tuple(head.parameters()), retain_graph=True)
                  for i in range(3)]
    projection, _ = project_distillation(gc, gk)
    for g, a, b, c in zip(got, gc, projection, gp):
        torch.testing.assert_close(g, a + b + c)


def test_ctc_and_preservation_keep_full_recording_when_teacher_has_edge_rows():
    head, v = phone_head(2, 3), example()
    lp = log_probs(head, v["features"], v["shipped"])
    full = objectives(lp, v["target"], v["teacher"], v["shipped"])
    short = objectives(lp, v["target"], v["teacher"][:-1], v["shipped"])
    assert torch.equal(full[0], short[0]) and torch.equal(full[2], short[2])
    with pytest.raises(ValueError, match="Impossible"):
        objectives(lp[:2], torch.tensor([1, 1]), v["teacher"][:2], v["shipped"][:2])


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
    from student_conflict_train import recordings
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
    from student_conflict_train import validate
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


def source_bundle(path):
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
        [h.make_tensor_value_info("log_probs", T.FLOAT, [1, 8, 3]), h.make_tensor_value_info("new_state", T.INT64, [1])],
        [n.from_array(np.array([1, 8, 80], dtype=np.int64), "shape"),
         n.from_array(np.random.default_rng(0).normal(0, .01, (80, 3)).astype(np.float32), "weight"),
         n.from_array(np.ones(1, dtype=np.int64), "one")])
    model = h.make_model(g, opset_imports=[h.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, path / "model.onnx")
    # Real external tensor storage exercises complete auxiliary bundle copying.
    teacher = h.make_model(h.make_graph([], "teacher", [], [],
        [n.from_array(np.ones(4, dtype=np.float32), "teacher_weight")]), ir_version=9)
    onnx.save_model(teacher, path / "teacher.onnx", save_as_external_data=True,
                   all_tensors_to_one_file=True, location="teacher.onnx.data", size_threshold=0)
    manifest = dict(file="model.onnx", sha256=digest(path / "model.onnx"), id="unit", tokens="ab", blank=0,
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
    new = graft(source, head, tmp_path / "export", "filtered")
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
        graft(source, head, tmp_path / "export", "filtered")
    (source / "teacher.onnx.data").unlink()
    with pytest.raises(ValueError, match="missing"):
        graft(source, head, tmp_path / "bad-export", "filtered")


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
    path = graft(source, head, tmp_path / "export", "filtered")
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
