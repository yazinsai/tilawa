"""Small synthetic tests; actual training and acoustic evaluation run on Modal."""

import ast
import copy
import io
import json
from pathlib import Path
import sys
import types

import pytest
import numpy as np
import soundfile as sf
import torch
from torch import nn

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent / "tools")]
import student_sequence as S

torch.set_num_threads(1)


def decoder(dropout=0.5):
    torch.manual_seed(7)
    return S.TranscriptionDecoder(6, 5, dim=8, heads=2, layers=1, history_dropout=dropout)


def test_reversal_is_per_recording_and_retains_repeated_phones():
    history, labels = S.ordered_targets([[1, 2, 2, 3], [4]], 5, "reverse", "cpu")
    assert history.tolist() == [[5, 3, 2, 2, 1], [5, 4, 7, 7, 7]]
    assert labels.tolist() == [[3, 2, 2, 1, 6], [4, 6, 7, 7, 7]]
    history, labels = S.ordered_targets([[1, 2, 2, 3], [4]], 5, "forward", "cpu")
    assert history[0].tolist() == [5, 1, 2, 2, 3]
    assert labels[0].tolist() == [1, 2, 2, 3, 6]


@pytest.mark.parametrize("sequences", [[], [[]], [[0]], [[5]], [[-1]], [[True]]])
def test_auxiliary_targets_refuse_blank_empty_and_noninventory_labels(sequences):
    with pytest.raises(ValueError):
        S.ordered_targets(sequences, 5, "reverse", "cpu")


def test_label_attention_is_causal_but_acoustic_attention_reaches_following_frames():
    model = decoder().eval()
    acoustic = torch.randn(1, 9, 6, requires_grad=True)
    history = torch.tensor([[5, 1, 2, 3]])
    logits = model(acoustic, torch.tensor([9]), history)
    changed = history.clone()
    changed[:, 2:] = torch.tensor([4, 4])
    assert torch.equal(logits[:, :2], model(acoustic, torch.tensor([9]), changed)[:, :2])
    logits[0, 0, 1].backward()
    assert acoustic.grad[0, -1].abs().sum() > 0
    assert model.acoustics.weight.grad.abs().sum() > 0


def test_padding_cannot_supervise_the_encoder():
    model = decoder().eval()
    acoustic = torch.randn(2, 9, 6, requires_grad=True)
    lens = torch.tensor([9, 4])
    loss = model.loss(acoustic, lens, [[1, 2, 3], [4]], "reverse")
    loss.backward()
    assert torch.isfinite(loss)
    assert acoustic.grad[1, 4:].abs().sum() == 0
    changed = acoustic.detach().clone()
    changed[1, 4:] = 1000
    assert torch.equal(loss.detach(), model.loss(changed, lens, [[1, 2, 3], [4]], "reverse"))


def test_complete_history_dropout_masks_phones_without_masking_bos_or_acoustics():
    model = decoder(1.0).train()
    acoustic = torch.randn(1, 6, 6)
    a, _ = S.ordered_targets([[1, 2, 3]], 5, "forward", "cpu")
    b, _ = S.ordered_targets([[4, 3, 2]], 5, "forward", "cpu")
    observed = []
    handle = model.embedding.register_forward_pre_hook(lambda _, args: observed.append(args[0].clone()))
    torch.manual_seed(9)
    result = model(acoustic, torch.tensor([6]), a)
    torch.manual_seed(9)
    alternative = model(acoustic, torch.tensor([6]), b)
    handle.remove()
    assert observed[0].tolist() == [[5, 8, 8, 8]]
    assert torch.equal(result, alternative)
    torch.manual_seed(9)
    assert not torch.equal(result, model(acoustic * 0, torch.tensor([6]), a))


def test_auxiliary_rng_is_private_and_continuation_supplies_no_gradient():
    model = decoder().train()
    acoustic = torch.randn(1, 6, 6, requires_grad=True)
    state = torch.get_rng_state().clone()
    reverse = S.auxiliary_loss(model, acoustic, torch.tensor([6]), [[1, 2]], "reverse", 23)
    assert torch.equal(state, torch.get_rng_state())
    reverse.backward()
    assert acoustic.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    acoustic.grad = None
    loss = S.auxiliary_loss(model, acoustic, torch.tensor([6]), [[1, 2]], "continuation", 23)
    loss.backward()
    assert loss == 0 and acoustic.grad.abs().sum() == 0
    assert all(p.grad is None for p in model.parameters())


class TinyPrimary(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(6, 6)
        self.encoder.chunk_size, self.encoder.left_context_frames = (16, 32), (64, 128)
        self.ctc_output = nn.Sequential(nn.Dropout(0.1), nn.Linear(6, 5))

    def streaming_eval(self, chunk, left):
        self.encoder.chunk_size, self.encoder.left_context_frames = (chunk,), (left,)

    def encode(self, feats, lens):
        return self.encoder(feats), lens

    def forward(self, feats, lens):
        encoded, lengths = self.encode(feats, lens)
        return self.ctc_output(encoded).log_softmax(-1), lengths


def batch():
    return dict(feats=torch.randn(2, 9, 6), feat_lens=torch.tensor([9, 6]),
                targets=torch.tensor([1, 2, 3, 4]), target_lens=torch.tensor([3, 1]),
                n_teacher=torch.tensor([9, 0]), top_ids=torch.tensor([[[1, 2, 3, 4, 0]] * 9] * 2),
                top_p=torch.full((2, 9, 5), 0.2), auxiliary=[[1, 2, 3], [4]])


def test_preservation_is_detached_and_includes_real_blank_frames():
    predicted = torch.randn(2, 9, 5).log_softmax(-1).detach().requires_grad_(True)
    anchor = torch.randn(2, 9, 5).log_softmax(-1).detach().requires_grad_(True)
    loss = S.preservation_loss(predicted, anchor, torch.tensor([9, 4]))
    loss.backward()
    assert anchor.grad is None
    assert predicted.grad[1, 4:].abs().sum() == 0
    assert predicted.grad[..., 0].abs().sum() > 0


def test_preservation_restores_mixed_training_flags_and_streaming_masks_on_error():
    model = TinyPrimary().train()
    model.ctc_output[0].eval()
    flags = [m.training for m in model.modules()]
    context = model.encoder.chunk_size, model.encoder.left_context_frames
    with pytest.raises(RuntimeError):
        with S.fixed_streaming(model, 16, 128):
            assert not model.training
            raise RuntimeError("synthetic failure")
    assert flags == [m.training for m in model.modules()]
    assert context == (model.encoder.chunk_size, model.encoder.left_context_frames)


def test_joint_training_retains_ctc_kd_and_trains_primary_from_auxiliary():
    torch.manual_seed(1)
    model = TinyPrimary().train()
    anchor = copy.deepcopy(model).eval().requires_grad_(False)
    aux = decoder().train()
    b = batch()
    cfg = dict(seed=5, chunk=16, left=128, aux_weight=0.2, preservation_weight=1.0)
    base = dict(ctc_weight=1.0, kd_weight=0.5, kd_delay=0)
    out = S.joint_objective(model, anchor, aux, b, base, cfg, "reverse", 1)
    out["loss"].backward()
    assert out["ctc"] > 0 and out["kd"] > 0 and out["auxiliary"] > 0
    assert model.encoder.weight.grad.abs().sum() > 0
    assert aux.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in anchor.parameters())
    assert model.training and model.encoder.chunk_size == (16, 32)


def test_zero_teacher_items_do_not_add_fake_kd():
    b = batch()
    b["n_teacher"].zero_()
    lp = torch.randn(2, 9, 5).log_softmax(-1).requires_grad_()
    primary, ctc, kd = S.primary_objective(lp, torch.tensor([9, 6]), b,
                                         dict(ctc_weight=1.0, kd_weight=0.5, kd_delay=0))
    assert kd == 0 and torch.equal(primary, ctc)
    primary.backward()
    assert torch.isfinite(lp.grad).all()


def test_new_transcripts_cannot_silently_zero_an_impossible_ctc_alignment():
    b = dict(targets=torch.tensor([1, 1, 1]), target_lens=torch.tensor([3]), n_teacher=torch.tensor([0]))
    lp = torch.randn(1, 3, 5).log_softmax(-1)
    with pytest.raises(ValueError, match="cannot align"):
        S.primary_objective(lp, torch.tensor([3]), b, dict(ctc_weight=1.0, kd_weight=0.0))


def test_selection_uses_primary_objective_and_all_valid_frames_and_restores_mode():
    model = TinyPrimary().train()
    anchor = copy.deepcopy(model).eval().requires_grad_(False)
    with torch.no_grad():
        model.ctc_output[-1].bias[1] += 0.8
    batches = [batch(), batch()]
    cfg = dict(chunk=16, left=128, preservation_weight=0.7)
    base = dict(ctc_weight=1.0, kd_weight=0.5, kd_delay=0)
    total, frames = 0.0, 0
    with torch.no_grad(), S.fixed_streaming(model, 16, 128):
        for b in batches:
            lp, lengths = model(b["feats"], b["feat_lens"])
            reference, _ = anchor(b["feats"], b["feat_lens"])
            primary, _, _ = S.primary_objective(lp, lengths, b, base)
            preserved = S.preservation_loss(lp, reference, lengths)
            n = int(lengths.sum())
            total += float(primary + cfg["preservation_weight"] * preserved) * n
            frames += n
    value = S.validation_objective(model, anchor, batches, base, cfg)
    assert value == pytest.approx(total / frames)
    assert model.training and model.encoder.chunk_size == (16, 32)
    assert not anchor.training
    # The validation metric must not read auxiliary labels or language loss.
    for b in batches:
        b["auxiliary"] = [[0], []]
    assert S.validation_objective(model, anchor, batches, base, cfg) == value


def test_production_student_encode_preserves_original_forward_and_parameter_keys(monkeypatch):
    # Execute the actual production class with small encoder doubles, avoiding
    # Modal/icefall imports locally. Real Zipformer/export parity still needs GPU.
    source = ast.parse((Path(__file__).parent / "student_train.py").read_text())
    factory = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "build_model")
    definition = next(n for n in factory.body if isinstance(n, ast.ClassDef) and n.name == "Student")

    class Embed(nn.Linear):
        def __init__(self, inp, dim, **kwargs):
            super().__init__(inp, dim)

        def forward(self, x, lengths):
            return super().forward(x[:, ::2]), (lengths + 1) // 2

    class Encoder(nn.Linear):
        def __init__(self, **kwargs):
            super().__init__(6, 6)
            self.chunk_size, self.left_context_frames = (16,), (128,)

        def forward(self, x, lengths, padding):
            assert padding.shape == (x.shape[1], x.shape[0])
            return super().forward(x[::2]), (lengths + 1) // 2

    utils = types.ModuleType("icefall.utils")
    utils.make_pad_mask = lambda lengths: torch.arange(int(lengths.max()))[None] >= lengths[:, None]
    monkeypatch.setitem(sys.modules, "icefall.utils", utils)
    keys = ("downsampling_factor", "num_encoder_layers", "encoder_unmasked_dim", "query_head_dim",
            "pos_head_dim", "value_head_dim", "num_heads", "feedforward_dim", "cnn_module_kernel",
            "chunk_size", "left_context_frames", "pos_dim")
    namespace = dict(nn=nn, dims=(6,), VOCAB=5, Conv2dSubsampling=Embed, Zipformer2=Encoder,
                     ScheduledFloat=lambda *a: 0.1, ints=lambda x: (int(x),), p=dict.fromkeys(keys, "1"))
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "student_train.py", "exec"), namespace)
    model = namespace["Student"]().train()
    feats, lens = torch.randn(2, 18, 80), torch.tensor([18, 14])
    rng = torch.get_rng_state().clone()
    embedded, embed_lens = model.encoder_embed(feats, lens)
    out, expected_lens = model.encoder(embedded.permute(1, 0, 2), embed_lens, utils.make_pad_mask(embed_lens))
    expected = model.ctc_output(out.permute(1, 0, 2)).log_softmax(-1)
    torch.set_rng_state(rng)
    actual, actual_lens = model(feats, lens)
    assert torch.equal(actual, expected) and torch.equal(actual_lens, expected_lens)
    assert set(model.state_dict()) == {"encoder_embed.weight", "encoder_embed.bias", "encoder.weight",
                                       "encoder.bias", "ctc_output.1.weight", "ctc_output.1.bias"}


def recipe():
    def item(key, speaker, source):
        return dict(key=key, speaker=speaker, source=source, original_split="train", auxiliary=True,
                    transcript="بِم", transcript_status="performed")
    return dict(schema=1, selection="minimum-primary-validation-objective",
                training_speakers=["q-train", "h-train"], validation_speakers=["q-valid", "h-valid"],
                excluded_speakers=["release"], anchor_sha256="a" * 64, corpus_sha256="b" * 64,
                training=[item("q:1", "q-train", "quran"), item("help:1", "h-train", "help-none")],
                validation=[item("q:2", "q-valid", "quran"), item("help:2", "h-valid", "help-none")],
                config=dict(epochs=1, batch_seconds=40, seed=0, validation_every=10, chunk=16, left=128,
                            aux_dim=8, aux_heads=2, aux_layers=1, learning_rate=0.0001,
                            aux_weight=0.2, preservation_weight=1.0, history_dropout=0.5))


def test_recipe_refuses_heldout_or_script_supervision_and_cross_speaker_partitions():
    record = recipe()
    S.recipe_contract(record)
    for edit in (lambda r: r["validation_speakers"].append("h-train"),
                 lambda r: r["training"][0].update(original_split="test"),
                 lambda r: r["training"][0].update(transcript_status="script"),
                 lambda r: r["config"].update(chunk=32),
                 lambda r: r["config"].update(history_dropout=0),
                 lambda r: r["validation"][1].update(key="help:1")):
        changed = copy.deepcopy(record)
        edit(changed)
        with pytest.raises(ValueError):
            S.recipe_contract(changed)


def test_none_selection_requires_explicit_clean_kind_and_original_training_split():
    rows = [dict(id="1", kind="none", split="train", speaker="h-train", scored=True,
                 events=[], non_errors=[], unsupported=[], ayahs=["1:1"])]
    selected = recipe()["training"]
    assert len(S.help_none_rows(rows, selected)) == 1
    for change in ({"kind": "repeat"}, {"split": "eval"}, {"scored": False},
                   {"non_errors": [{"kind": "repeat"}]}, {"unsupported": ["vowel"]},
                   {"excluded": "review"}, {"speaker": "release"}):
        with pytest.raises(ValueError):
            S.help_none_rows([{**rows[0], **change}], selected)


def test_audited_loader_retains_complete_audio_and_mixture_and_detects_drift(tmp_path, monkeypatch):
    record = recipe()
    buf = io.BytesIO()
    sf.write(buf, np.linspace(-0.2, 0.2, 1600), 16000, format="FLAC", subtype="PCM_16")
    audio = buf.getvalue()
    clips = dict(key=["q:1", "q:2"], flac=np.frombuffer(audio * 2, dtype=np.uint8),
                 flac_off=np.array([0, len(audio), 2 * len(audio)]), top_off=np.array([0, 5, 10]),
                 ids=np.array([1, 2, 3, 4, 0] * 2), probs=np.full(10, 0.2),
                 ref=["بِم"] * 2, hyp=["بِم"] * 2, n_samples=np.array([1600, 1600]))
    calls = []
    train = types.ModuleType("student_train")
    train.SR = 16000
    train.load_clips = lambda *args: calls.append(args) or clips
    monkeypatch.setitem(sys.modules, "student_train", train)
    base = dict(buckets="clean-match", max_hours=0.0, max_clip_s=30.0, seed=7, target="teacher")
    labels = []
    for i, speaker in ((1, "h-train"), (2, "h-valid")):
        sf.write(tmp_path / f"{i}.wav", np.linspace(-0.2, 0.2, 1600), 16000, subtype="PCM_16")
        labels.append(dict(id=str(i), kind="none", split="train", speaker=speaker, scored=True,
                           events=[], non_errors=[], unsupported=[], ayahs=["1:1"], wav=f"{i}.wav"))
    (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels))
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(dict(ayahs=[dict(s=1, a=1, p="بِم")])))
    audit = S.audit_corpus(record, base, tmp_path, reference)
    record["corpus_sha256"] = audit["corpus_sha256"]
    rows, valid = S.load_recordings(record, base, tmp_path, reference)
    assert calls == [("clean-match", "train", 0.0, 30.0, 7)] * 2
    assert len(rows) == len(valid) == 2
    help_row = next(r for r in rows if r["key"] == "help:1")
    assert help_row["audio"] == (tmp_path / "1.wav").read_bytes()
    assert help_row["samples"] == 1600 and help_row["auxiliary"] == help_row["target"]
    assert help_row["top_ids"].shape == help_row["top_p"].shape == (0, 5)
    changed = copy.deepcopy(record)
    changed["training"].pop(0)
    with pytest.raises(ValueError):
        S.load_recordings(changed, base, tmp_path, reference)
    clips["probs"][0] = 0.3
    with pytest.raises(ValueError, match="differs from its frozen audit"):
        S.load_recordings(record, base, tmp_path, reference)
    clips["probs"][0] = 0.2
    sf.write(tmp_path / "1.wav", np.zeros(1600), 16000, subtype="PCM_16")
    with pytest.raises(ValueError, match="differs from its frozen audit"):
        S.load_recordings(record, base, tmp_path, reference)


def test_deployment_checkpoint_discards_decoder_and_roundtrips_original_primary_loader_shape(tmp_path):
    model, aux = TinyPrimary(), decoder()
    selected = S.primary_checkpoint(model, {"size": "tiny"}, dict(step=4))
    assert selected["model"].keys() == model.state_dict().keys()
    assert all("auxiliary" not in k and "decoder" not in k for k in selected["model"])
    path = tmp_path / "ckpt.pt"
    torch.save(selected, path)
    saved = torch.load(path, weights_only=False)
    restored = TinyPrimary()
    restored.load_state_dict(saved["model"], strict=True)
    x, lens = torch.randn(1, 9, 6), torch.tensor([9])
    model.eval()
    restored.eval()
    assert torch.equal(model(x, lens)[0], restored(x, lens)[0])
    aux.output.weight.data.add_(1)
    assert torch.equal(model(x, lens)[0], restored(x, lens)[0])
