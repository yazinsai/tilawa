import copy
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student_deletion import admitted_none, speaker_partition, deletion_targets, sequence_log_prob, weak_contrast, phone_head, log_probs, objectives
from student_deletion_export import graft, expose, session, stream, digest


def exhaustive_ctc(log_probs, target):
    paths = []
    for path in itertools.product(range(log_probs.shape[1]), repeat=len(log_probs)):
        collapsed = [k for k, _ in itertools.groupby(path) if k]
        if collapsed == target:
            paths.append(sum(log_probs[i, k] for i, k in enumerate(path)))
    return torch.logsumexp(torch.stack(paths), 0)


@pytest.mark.parametrize("target", [[], [1], [1, 1], [1, 2], [2, 1, 2]])
def test_forward_sum_and_gradient_match_exhaustive_ctc_paths(target):
    logits = torch.randn(4, 3, generator=torch.Generator().manual_seed(7), dtype=torch.float64,
                         requires_grad=True)
    lp = logits.log_softmax(-1)
    got = sequence_log_prob(lp, [torch.tensor(target, dtype=torch.int64)])[0]
    want = exhaustive_ctc(lp, target)
    torch.testing.assert_close(got, want)
    actual_gradient, = torch.autograd.grad(got, logits, retain_graph=True)
    expected_gradient, = torch.autograd.grad(want, logits)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_each_negative_removes_one_consonant_occurrence_and_keeps_vowels():
    tokens = "بَتُ"
    target = torch.tensor([1, 2, 3, 4, 1], dtype=torch.int64)
    positions, negatives = deletion_targets(target, tokens)
    assert positions == [0, 2, 4]
    assert [n.tolist() for n in negatives] == [[2, 3, 4, 1], [1, 2, 4, 1], [1, 2, 3, 4]]
    positions, negatives = deletion_targets(torch.tensor([1, 1, 2]), tokens)
    assert positions == [0, 1]
    assert [n.tolist() for n in negatives] == [[1, 2], [1, 2]]
    assert deletion_targets(torch.tensor([2, 4]), tokens) == ([], [])


def test_contrast_normalizes_occurrences_and_emphasizes_weakest_comparison():
    margins = torch.tensor([-2., 10., 10.], dtype=torch.float64, requires_grad=True)
    loss = weak_contrast(margins)
    torch.testing.assert_close(loss, weak_contrast(margins.repeat(7)))
    gradient, = torch.autograd.grad(loss, margins)
    assert abs(gradient[0]) > 1000 * abs(gradient[1])
    torch.testing.assert_close(weak_contrast(torch.zeros(9, dtype=torch.float64)),
                               torch.tensor(math.log(2), dtype=torch.float64))
    assert torch.isfinite(weak_contrast(torch.tensor([-10000., 10000.])))
    empty = torch.empty(0, requires_grad=True)
    weak_contrast(empty).backward()
    assert empty.grad is not None


def test_strong_shared_vowel_does_not_hide_a_dispensable_consonant():
    target = torch.tensor([1, 2])
    weak = torch.tensor([[.90, .01, .09], [.01, .01, .98]]).log()
    strong = torch.tensor([[.01, .90, .09], [.01, .01, .98]]).log()
    _, _, weak_loss = objectives(weak, target, weak, "بَ", contrast=True)
    _, _, strong_loss = objectives(strong, target, strong, "بَ", contrast=True)
    assert weak_loss > 3 and strong_loss < .2


def test_ordinary_arm_is_plain_ctc_with_identical_preservation_and_frozen_encoder():
    features = torch.randn(5, 4, requires_grad=True)
    shipped = torch.randn(5, 3, requires_grad=True)
    head = phone_head(4, 3)
    prediction = log_probs(head, features, shipped)
    target = torch.tensor([1, 2])
    ordinary = objectives(prediction, target, shipped, "بَ")
    contrast = objectives(prediction, target, shipped, "بَ", contrast=True)
    expected_ctc = -sequence_log_prob(prediction, [target])[0] / len(target)
    torch.testing.assert_close(ordinary[0], expected_ctc)
    torch.testing.assert_close(ordinary[0], contrast[0])
    torch.testing.assert_close(ordinary[1], contrast[1])
    assert ordinary[2] == 0
    torch.testing.assert_close(prediction, shipped.detach().log_softmax(-1))
    sum(contrast).backward()
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert features.grad is None and shipped.grad is None


@pytest.mark.parametrize("target", [torch.tensor([0]), torch.tensor([3]), torch.tensor([1, 1, 1]),
                                   torch.tensor([1.])])
def test_invalid_or_impossible_transcripts_fail(target):
    with pytest.raises(ValueError):
        sequence_log_prob(torch.zeros(3, 3).log_softmax(-1), [target])
    if target.dtype != torch.int64 or (target <= 0).any() or (target > 2).any():
        with pytest.raises(ValueError):
            deletion_targets(target, "بَ")


def test_selection_is_speaker_balanced_and_read_only():
    from student import TOKENS
    from student_deletion_train import validate
    head = phone_head(4, len(TOKENS) + 1)
    def row(speaker, bias):
        logits = torch.zeros(5, len(TOKENS) + 1)
        logits[:, 1] = bias
        view = dict(features=torch.zeros(5, 4), shipped=logits.log_softmax(-1), target=torch.tensor([1]))
        return dict(speaker=speaker, views=[view])
    a, b = row("many-takes", 5.), row("one-take", -2.)
    before = copy.deepcopy(head.state_dict())
    one, two = validate(head, [a]), validate(head, [b])
    actual = validate(head, [a, a, a, b])
    assert actual["speakers"] == 2
    for key in ("objective", "per"):
        assert actual[key] == pytest.approx((one[key] + two[key]) / 2)
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, before[key])

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
    from student_deletion_train import recordings
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
    new = graft(source, head, tmp_path / "export", "contrast")
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
        graft(source, head, tmp_path / "export", "contrast")
    (source / "teacher.onnx.data").unlink()
    with pytest.raises(ValueError, match="missing"):
        graft(source, head, tmp_path / "bad-export", "contrast")


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
    path = graft(source, head, tmp_path / "export", "contrast")
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


def test_matched_fit_exports_both_selected_checkpoints_from_cached_features(tmp_path, monkeypatch):
    import argparse
    import student_deletion_train as trainer
    from student import TOKENS
    source = tmp_path / "source"
    source_bundle(source, TOKENS)
    prepared = tmp_path / "prepared"
    m, feature, width, extraction = expose(source, prepared)
    banks = np.zeros((45, 80), dtype=np.float32)
    shipped, features = stream(session(extraction), m, banks, feature)
    rows = []
    for speaker in ("a", "b", "c", "d", "e"):
        for corpus in ("help-none", "transcribed"):
            files = []
            for v in range(3):
                file = f"{speaker}-{corpus}-{v}.npz"
                np.savez_compressed(prepared / file, features=features, shipped=shipped,
                                    target=np.array([1, 2], dtype=np.int64), fbank=banks)
                files.append(file)
            rows.append(dict(speaker=speaker, source=corpus, views=files,
                             viewHashes=[digest(prepared / file) for file in files]))
    trainer.write_json(prepared / "prepared.json", dict(schema=1, source=str(source),
        sourceSha256=m["sha256"], tokens=TOKENS, width=width, rows=rows,
        groups=speaker_partition([r["speaker"] for r in rows]), excludedSpeakers=[],
        auxiliaryHashes=trainer.insertion_assets(source)))
    monkeypatch.setattr(trainer, "EPOCHS", 1)
    out = tmp_path / "fit"
    trainer.fit(argparse.Namespace(prepared=str(prepared), out=str(out)))
    report = json.loads((out / "training-report.json").read_text())
    assert set(report["arms"]) == {"ordinary", "contrast"}
    assert report["arms"]["ordinary"]["history"][0] == report["arms"]["contrast"]["history"][0]
    for arm, result in report["arms"].items():
        assert result["selectedEpoch"] in (0, 1)
        assert result["parityRecordings"] == 2
        assert digest(out / arm / "model.onnx") == result["sha256"]
        assert (out / (arm + ".pt")).is_file()
    # A tampered cache cannot be reused for a refit.
    (prepared / rows[0]["views"][0]).write_bytes(b"altered")
    with pytest.raises(ValueError, match="Cached training evidence changed"):
        trainer.load_prepared(prepared)
