import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_duration import admitted_none, speaker_partition, ctc_occupancy, identity_weights, identity_budget, phone_head, log_probs, objectives, recording_loss
from student_duration_export import graft, expose, session, stream, digest, insertion_assets

torch.set_num_threads(1)


def example():
    x = torch.tensor([[1., .1], [.5, .2], [.2, .7], [.1, 1.], [.2, .4], [.1, .1]])
    shipped = torch.tensor([[3., 0., 0.]]).repeat(len(x), 1).log_softmax(-1)
    target = torch.tensor([1, 2])
    occupancy, _, _ = ctc_occupancy(shipped, target)
    return dict(features=x, shipped=shipped, teacher=shipped.clone(), target=target, occupancy=occupancy)


@pytest.mark.parametrize("target", [[1], [1, 2], [1, 1], [1, 2, 1]])
def test_ctc_marginals_match_exhaustive_label_paths(target):
    import itertools
    # Enumerating emitted phone sequences is independent of state recursion.
    lp = torch.tensor([[.2, .6, .2], [.5, .2, .3], [.2, .5, .3],
                       [.6, .1, .3], [.1, .7, .2]], dtype=torch.float64).log()
    q, blank, log_z = ctc_occupancy(lp.requires_grad_(), torch.tensor(target))
    want, want_blank, total = torch.zeros_like(q), torch.zeros_like(blank), 0.
    for path in itertools.product(range(3), repeat=len(lp)):
        collapsed, occurrence, indices, previous = [], -1, [], None
        for label in path:
            if label and label != previous:
                collapsed.append(label)
                occurrence += 1
            indices.append(occurrence if label else None)
            previous = label
        if collapsed != target:
            continue
        probability = float(sum(lp[t, label].detach() for t, label in enumerate(path)).exp())
        total += probability
        for t, j in enumerate(indices):
            if j is None:
                want_blank[t] += probability
            else:
                want[t, j] += probability
    torch.testing.assert_close(q, want / total, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(blank, want_blank / total, atol=1e-12, rtol=1e-12)
    assert float(log_z.exp()) == pytest.approx(total)
    assert not q.requires_grad and not blank.requires_grad


def test_short_and_sustained_repeated_occurrences_get_equal_identity_mass():
    # Same phone twice: one frame, a separating blank, then five frames.
    lp = torch.full((7, 3), -torch.inf, dtype=torch.float64)
    lp[[0, 2, 3, 4, 5, 6], 1] = 0
    lp[1, 0] = 0
    q, blank, _ = ctc_occupancy(lp, torch.tensor([1, 1]))
    assert q.sum(0).tolist() == [1., 5.]
    assert blank.tolist() == [0., 1., 0., 0., 0., 0., 0.]
    occurrence, frame = identity_weights(q, "occurrence"), identity_weights(q, "frame")
    torch.testing.assert_close(occurrence.sum(0), torch.tensor([.5, .5], dtype=q.dtype))
    torch.testing.assert_close(frame.sum(0), torch.tensor([1/6, 5/6], dtype=q.dtype))
    assert float(occurrence.sum()) == pytest.approx(float(frame.sum()))


def test_long_low_probability_transcript_alignment_is_stable():
    lp = torch.tensor([0., -80., -100.]).repeat(400, 1)
    q, blank, log_z = ctc_occupancy(lp, torch.tensor([1, 2] * 10))
    assert torch.isfinite(q).all() and torch.isfinite(log_z)
    torch.testing.assert_close(q.sum(1) + blank, torch.ones(400, dtype=torch.float64))
    torch.testing.assert_close(identity_weights(q, "occurrence").sum(0),
                               torch.full((20,), .05, dtype=torch.float64))


def test_corpus_budget_preserves_equal_occurrence_mass_across_recording_lengths():
    short = example()
    long = {**short, "target": torch.tensor([1, 2, 1]),
            "occupancy": ctc_occupancy(short["teacher"], torch.tensor([1, 2, 1]))[0]}
    budget = identity_budget([dict(views=[short]), dict(views=[long])])
    masses = [identity_weights(v["occupancy"], "occurrence", budget["occurrence"]).sum(0)
              for v in (short, long)]
    torch.testing.assert_close(torch.cat(masses), torch.full((5,), .4, dtype=torch.float64))
    for arm in ("frame", "occurrence"):
        total = sum(float(identity_weights(v["occupancy"], arm, budget[arm]).sum())
                    for v in (short, long)) / 2
        assert total == pytest.approx(1.)


@pytest.mark.parametrize("target", [[0], [3], [1, 1, 1]])
def test_invalid_or_unalignable_transcript_fails(target):
    with pytest.raises(ValueError, match="Impossible"):
        ctc_occupancy(torch.zeros(3, 3), torch.tensor(target))


def test_zero_probability_transcript_fails_instead_of_skipping_recording():
    lp = torch.tensor([[0., -torch.inf, -torch.inf]]).repeat(3, 1)
    with pytest.raises(ValueError, match="zero probability"):
        ctc_occupancy(lp, torch.tensor([1]))


def test_blank_ctc_preservation_unchanged_and_identity_cannot_push_blank():
    v = example()
    # Artificial occupancy makes duration imbalance large and deterministic.
    v["occupancy"] = torch.tensor([[1., 0.], [0., 1.], [0., 1.],
                                    [0., 1.], [0., 1.], [0., 0.]], requires_grad=True)
    logits = torch.tensor([[1., 2., -1.], [1., -.2, .3], [1., -.1, 1.],
                           [1., 2., 0.], [1., .5, -.5], [2., 0., 0.]], requires_grad=True)
    lp = logits.log_softmax(-1)
    frame = objectives(lp, v["target"], v["teacher"], v["shipped"], v["occupancy"], "frame")
    occurrence = objectives(lp, v["target"], v["teacher"], v["shipped"], v["occupancy"], "occurrence")
    for i in (0, 2, 3):
        assert torch.equal(frame[i], occurrence[i])
        a = torch.autograd.grad(frame[i], logits, retain_graph=True)[0]
        b = torch.autograd.grad(occurrence[i], logits, retain_graph=True)[0]
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.equal(frame[1], occurrence[1])
    for identity in (frame[1], occurrence[1]):
        gradient = torch.autograd.grad(identity, logits, retain_graph=True)[0]
        torch.testing.assert_close(gradient[:, 0], torch.zeros(6), atol=1e-7, rtol=0)
    assert v["occupancy"].grad is None


def test_real_student_tail_keeps_ctc_and_preservation_and_evidence_is_frozen():
    head, v = phone_head(2, 3), example()
    for k in ("features", "shipped", "teacher"):
        v[k].requires_grad_(True)
    full = objectives(log_probs(head, v["features"], v["shipped"]), v["target"],
                      v["teacher"], v["shipped"], v["occupancy"], "frame")
    w = {**v, "teacher": v["teacher"][:-1],
         "occupancy": ctc_occupancy(v["teacher"][:-1], v["target"])[0]}
    short = objectives(log_probs(head, w["features"], w["shipped"]), w["target"],
                       w["teacher"], w["shipped"], w["occupancy"], "occurrence")
    assert torch.equal(full[0], short[0]) and torch.equal(full[3], short[3])
    loss, _ = recording_loss(head, [v, w], "occurrence")
    loss.backward()
    assert all(v[k].grad is None for k in ("features", "shipped", "teacher"))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


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
    from student_duration_train import recordings
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
    from student_duration_train import validate
    head, first = phone_head(2, 3), example()
    second = {**first, "target": torch.tensor([2, 1, 2]),
              "occupancy": ctc_occupancy(first["teacher"], torch.tensor([2, 1, 2]))[0]}
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
    nodes = [h.make_node("Slice", ["x", "start", "end", "axis"], ["committed"]),
             h.make_node("ReduceMean", ["committed"], ["mean"], axes=[1], keepdims=1),
             h.make_node("Expand", ["mean", "shape"], ["feature"]),
             h.make_node("MatMul", ["feature", "weight"], ["scores"], name="ctc_output/MatMul"),
             h.make_node("LogSoftmax", ["scores"], ["log_probs"], axis=-1),
             h.make_node("Add", ["state", "one"], ["new_state"])]
    g = h.make_graph(nodes, "test", [h.make_tensor_value_info("x", T.FLOAT, [1, 45, 80]),
        h.make_tensor_value_info("state", T.INT64, [1])],
        [h.make_tensor_value_info("log_probs", T.FLOAT, [1, 8, len(tokens) + 1]), h.make_tensor_value_info("new_state", T.INT64, [1])],
        [n.from_array(np.array([0], dtype=np.int64), "start"),
         n.from_array(np.array([32], dtype=np.int64), "end"),
         n.from_array(np.array([1], dtype=np.int64), "axis"),
         n.from_array(np.array([1, 8, 80], dtype=np.int64), "shape"),
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


def test_matched_fit_selects_and_exports_same_update_count_without_touching_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import student_duration_train as training
    from student import TOKENS
    source, prepared, out = (tmp_path / name for name in ("source", "prepared", "fit"))
    manifest = source_bundle(source, TOKENS)
    m, feature, width, extractor = expose(source, prepared)
    runtime, rows = session(extractor), []
    for i in range(10):
        f = np.random.default_rng(i).normal(.2, .5, (99, 80)).astype(np.float32)
        lp, x = stream(runtime, m, f, feature)
        target = np.array([1, 2, 1], dtype=np.int64)
        occupancy, _, _ = ctc_occupancy(torch.tensor(lp), torch.tensor(target))
        files = []
        for v in range(3):
            name = f"recording-{i}-view-{v}.npz"
            np.savez_compressed(prepared / name, features=x, shipped=lp, target=target,
                                teacher=lp, occupancy=occupancy.numpy(), fbank=f)
            files.append(name)
        rows.append(dict(speaker=f"speaker-{i // 2}", source="help-none" if i % 2 else "transcribed",
                         views=files, viewHashes=[digest(prepared / file) for file in files]))
    groups = speaker_partition([r["speaker"] for r in rows])
    info = dict(source=str(source), sourceSha256=m["sha256"], tokens=TOKENS, width=width,
                rows=rows, groups=groups, excludedSpeakers=[], auxiliaryHashes=insertion_assets(source))
    training.write_json(prepared / "prepared.json", info)
    before = {p.name: digest(p) for p in prepared.iterdir()}
    monkeypatch.setattr(training, "EPOCHS", 2)
    training.fit(SimpleNamespace(prepared=str(prepared), out=str(out)))
    report = json.loads((out / "training-report.json").read_text())
    a, b = report["arms"]["frame"], report["arms"]["occurrence"]
    assert a["trainingUpdates"] == b["trainingUpdates"] == 16
    assert a["selectedEpoch"] == b["selectedEpoch"] == report["selectedEpoch"]
    assert report["selectedEpoch"] > 0
    assert a["selectedUpdates"] == b["selectedUpdates"] == report["selectedEpoch"] * 8
    assert len(a["history"]) == len(b["history"]) == 3
    assert a["parityRecordings"] == b["parityRecordings"] == 2
    assert before == {p.name: digest(p) for p in prepared.iterdir()}
    assert digest(source / "model.onnx") == manifest["sha256"]
    assert insertion_assets(out / "frame") == insertion_assets(out / "occurrence") == insertion_assets(source)


def test_export_preserves_caches_clock_assets_and_selected_head(tmp_path):
    source = tmp_path / "source"
    manifest = source_bundle(source)
    m, feature, width, extractor = expose(source, tmp_path / "extraction")
    assert width == 80
    head = phone_head(width, 3)
    with torch.no_grad():
        head.bias.copy_(torch.tensor([.2, -.1, .3]))
        head.weight[:, :2] = torch.tensor([[.1, .2], [.3, -.1], [0., .2]])
    new = graft(source, head, tmp_path / "export", "occurrence")
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
        graft(source, head, tmp_path / "export", "occurrence")
    (source / "teacher.onnx.data").unlink()
    with pytest.raises(ValueError, match="missing"):
        graft(source, head, tmp_path / "bad-export", "occurrence")


def test_zero_head_export_preserves_distribution(tmp_path):
    source = tmp_path / "source"
    m = source_bundle(source)
    path = graft(source, phone_head(80, 3), tmp_path / "export", "frame")
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
    path = graft(source, head, tmp_path / "export", "occurrence")
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
