"""Pure helpers for icefall Zipformer-CTC export (no k2 / icefall / GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from zipformer_ctc_utils import (  # noqa: E402
    ARCH_FLAGS,
    DEFAULT_AVG,
    DEFAULT_BASE_LR,
    DEFAULT_INIT_FROM,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_TRAIN_CHUNK_SIZES,
    DEFAULT_TRAIN_SOURCES,
    VOCAB_SIZE,
    build_metadata,
    compute_T_hop,
    detect_state_space,
    icefall_to_ref_ids,
    icefall_train_flags,
    inverse_permute_ctc_head,
    io_inputs_match,
    io_json_from_session,
    json_plain,
    missing_fbank_sources,
    parse_source_weights,
    permute_ctc_head,
    ref_to_icefall_ids,
    remap_quranlab_key,
    resolve_train_sources,
    scale_mux_weights,
    write_icefall_tokens,
    interpolate_state_dicts,
    parse_export_interp,
    select_checkpoint_state,
    CTC_ICEFALL_BIAS_KEY,
    CTC_ICEFALL_WEIGHT_KEY,
)

REF_IO = ROOT / "experiments" / "zipformer-ctc" / "zipformer-io.json"
TOKENS_TXT = ROOT / "experiments" / "zipformer-ctc" / "tokens.txt"


def test_ref_icefall_id_roundtrip_blank_250_to_0():
    ids = np.arange(VOCAB_SIZE, dtype=np.int64)
    ice = ref_to_icefall_ids(ids)
    assert ice[250] == 0
    assert ice[0] == 1
    assert ice[249] == 250
    back = icefall_to_ref_ids(ice)
    np.testing.assert_array_equal(back, ids)
    # list / scalar
    assert ref_to_icefall_ids(250) == 0
    assert icefall_to_ref_ids(0) == 250
    assert icefall_to_ref_ids(ref_to_icefall_ids([3, 250, 0])) == [3, 250, 0]


def test_permute_ctc_head_softmax_alignment():
    rng = np.random.default_rng(0)
    hidden = 32
    x = rng.standard_normal((7, hidden)).astype(np.float32)
    w = rng.standard_normal((VOCAB_SIZE, hidden)).astype(np.float32)
    b = rng.standard_normal(VOCAB_SIZE).astype(np.float32)
    w_perm, b_perm = permute_ctc_head(w, b)

    def softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    p_ice = softmax(x @ w.T + b)
    p_ref = softmax(x @ w_perm.T + b_perm)
    for ref_id in range(VOCAB_SIZE):
        ice_id = int(ref_to_icefall_ids(ref_id))
        np.testing.assert_allclose(
            p_ref[:, ref_id], p_ice[:, ice_id], rtol=1e-5, atol=1e-6
        )


def test_inverse_permute_ctc_head_roundtrip_and_blank_row():
    rng = np.random.default_rng(1)
    hidden = 8
    w = rng.standard_normal((VOCAB_SIZE, hidden)).astype(np.float32)
    b = rng.standard_normal(VOCAB_SIZE).astype(np.float32)
    w_back, b_back = permute_ctc_head(*inverse_permute_ctc_head(w, b))
    np.testing.assert_array_equal(w_back, w)
    np.testing.assert_array_equal(b_back, b)
    w_back, b_back = inverse_permute_ctc_head(*permute_ctc_head(w, b))
    np.testing.assert_array_equal(w_back, w)
    np.testing.assert_array_equal(b_back, b)

    marked_w = np.zeros((VOCAB_SIZE, 3), dtype=np.float32)
    marked_b = np.zeros(VOCAB_SIZE, dtype=np.float32)
    marked_w[250] = 3.0
    marked_b[250] = 7.0
    ice_w, ice_b = inverse_permute_ctc_head(marked_w, marked_b)
    np.testing.assert_array_equal(ice_w[0], 3.0)
    assert ice_b[0] == 7.0
    assert ice_b[250] == 0.0


def test_remap_quranlab_keys_sub_and_ctc_drop_transducer():
    assert remap_quranlab_key("sub.conv.0.weight") == "encoder_embed.conv.0.weight"
    assert remap_quranlab_key("ctc_head.weight") == "ctc_output.1.weight"
    assert remap_quranlab_key("ctc_head.bias") == "ctc_output.1.bias"
    assert remap_quranlab_key("encoder.encoders.0.layers.0.norm.bias") == (
        "encoder.encoders.0.layers.0.norm.bias"
    )
    assert remap_quranlab_key("decoder.embedding.weight") is None
    assert remap_quranlab_key("joiner.output_linear.weight") is None
    assert remap_quranlab_key("simple_am_proj.weight") is None
    assert remap_quranlab_key("simple_lm_proj.bias") is None


def test_icefall_train_flags_init_from_chunk_lr():
    flags = icefall_train_flags(init_from=DEFAULT_INIT_FROM)
    assert flags[flags.index("--chunk-size") + 1] == DEFAULT_TRAIN_CHUNK_SIZES
    assert flags[flags.index("--chunk-size") + 1] == "8,16,24"
    assert flags[flags.index("--left-context-frames") + 1] == "128,256"
    assert flags[flags.index("--base-lr") + 1] == str(DEFAULT_BASE_LR)
    assert flags[flags.index("--base-lr") + 1] == "0.005"
    assert flags[flags.index("--num-epochs") + 1] == str(DEFAULT_NUM_EPOCHS)
    assert flags[flags.index("--drop-last") + 1] == "1"
    limited = icefall_train_flags(limit_cuts=800, sources="retasy")
    assert limited[limited.index("--drop-last") + 1] == "0"
    assert limited[limited.index("--num-buckets") + 1] == "4"
    assert limited[limited.index("--sources") + 1] == "retasy"
    train_py = (ROOT / "scripts" / "train_zipformer_ctc_modal.py").read_text()
    assert "init_from: str = DEFAULT_INIT_FROM" in train_py
    assert train_py.count("init_from: str = DEFAULT_INIT_FROM") >= 2
    assert "arch_train_flags" in train_py
    assert "icefall_train_flags" in train_py
    assert "icefall_state_from_reference" in train_py
    assert "0.005" in train_py
    assert DEFAULT_AVG == 3
    assert DEFAULT_NUM_EPOCHS == 5
    assert "TypeError" in train_py
    assert "LHOTSE_FBANK_CONFIG" in train_py
    assert "torchaudio_compatible_mel_scale" in train_py
    assert "cfg.pop(" not in train_py
    ddp = icefall_train_flags(world_size=4)
    assert ddp[ddp.index("--world-size") + 1] == "4"
    assert "torch.cuda.device_count" in train_py
    assert 'env={"ZIPFORMER_GPU": _GPU_SPEC}' in train_py


def test_parse_source_weights_and_mux_scale():
    assert parse_source_weights("") == {}
    assert parse_source_weights("   ") == {}
    assert parse_source_weights("everyayah_multi=2.5") == {"everyayah_multi": 2.5}
    assert parse_source_weights("a=1.5,b=2") == {"a": 1.5, "b": 2.0}
    assert parse_source_weights(" a = 1.5 , b=2 ") == {"a": 1.5, "b": 2.0}
    hours = [358.0, 179.4, 747.2]
    srcs = ["everyayah", "everyayah_multi", "qua"]
    scaled = scale_mux_weights(
        hours, srcs, parse_source_weights("everyayah_multi=2.5")
    )
    assert scaled == pytest.approx([358.0, 448.5, 747.2])
    # ~25% of the mux mass (hours × override), vs ~11% unweighted.
    assert scaled[1] / sum(scaled) == pytest.approx(448.5 / (358.0 + 448.5 + 747.2))
    assert scale_mux_weights(hours, srcs, {}) == hours
    unknown = scale_mux_weights(hours, srcs, {"not_in_mix": 9.0})
    assert unknown == hours
    with pytest.raises(ValueError, match="source=factor"):
        parse_source_weights("noequals")
    with pytest.raises(ValueError, match="not a float"):
        parse_source_weights("a=xyz")
    with pytest.raises(ValueError, match=">= 0"):
        parse_source_weights("a=-1")
    with pytest.raises(ValueError, match="duplicate"):
        parse_source_weights("a=1,a=2")
    with pytest.raises(ValueError, match="length mismatch"):
        scale_mux_weights([1.0], ["a", "b"], {})
    flags = icefall_train_flags(source_weights="everyayah_multi=2.5")
    assert flags[flags.index("--source-weights") + 1] == "everyayah_multi=2.5"
    default_flags = icefall_train_flags()
    assert default_flags[default_flags.index("--source-weights") + 1] == ""
    train_py = (ROOT / "scripts" / "train_zipformer_ctc_modal.py").read_text()
    data_py = (ROOT / "scripts" / "zipformer_asr_datamodule.py").read_text()
    assert train_py.count("source_weights: str = \"\"") == 2
    assert "source_weights=source_weights" in train_py
    assert "--source-weights" in data_py
    assert "scale_mux_weights" in data_py
    assert "parse_source_weights" in data_py


def test_compute_T_hop_reference_chunk_24():
    assert compute_T_hop(24) == (61, 48)
    t, hop = compute_T_hop(16)
    assert t == 16 * 2 + 13
    assert hop == 32


def test_write_icefall_tokens_blk_first_roundtrip(tmp_path: Path):
    sys.path.insert(0, str(ROOT))
    from shared.phoneme_labels import load_tokens

    tokens = load_tokens(TOKENS_TXT)
    path = tmp_path / "tokens_icefall.txt"
    write_icefall_tokens(tokens, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "<blk> 0"
    assert tokens[-1] == "<blank>"
    assert f"{tokens[0]} 1" == lines[1]
    assert f"{tokens[249]} 250" == lines[-1]
    assert len(lines) == VOCAB_SIZE
    parsed = {}
    for line in lines:
        sym, idx = line.rsplit(" ", 1)
        parsed[int(idx)] = sym
    assert parsed[0] == "<blk>"
    assert 250 in parsed
    assert "<blank>" not in parsed.values()


def test_io_json_from_session_matches_reference_schema():
    ref = json.loads(REF_IO.read_text(encoding="utf-8"))
    inputs_meta = [
        SimpleNamespace(name=inp["name"], shape=list(inp["dims"]), type="tensor(float)")
        if inp["dtype"] == "float32"
        else SimpleNamespace(
            name=inp["name"], shape=list(inp["dims"]), type="tensor(int64)"
        )
        for inp in ref["inputs"]
    ]
    outputs_meta = [
        SimpleNamespace(name="log_probs", shape=[1, 24, 251], type="tensor(float)"),
        SimpleNamespace(
            name="new_embed_states", shape=[1, 128, 3, 19], type="tensor(float)"
        ),
        SimpleNamespace(name="new_processed_lens", shape=[1], type="tensor(int64)"),
    ]
    built = io_json_from_session(
        inputs_meta,
        outputs_meta,
        model=ref["model"],
        T=ref["T"],
        hop=ref["hop"],
        feature_dim=ref["featureDim"],
        vocab_size=ref["vocabSize"],
    )
    for key in ref:
        assert key in built, f"missing key {key}"
    assert built["T"] == 61
    assert built["hop"] == 48
    assert built["featureDim"] == 80
    assert built["vocabSize"] == 251
    assert {i["name"] for i in built["inputs"]} == {i["name"] for i in ref["inputs"]}
    for a, b in zip(built["inputs"], ref["inputs"]):
        assert a["name"] == b["name"]
        assert a["dims"] == b["dims"]
        assert a["dtype"] == b["dtype"]
    assert built["outputs"][0]["dtype"] == "float32"
    assert built["outputs"][-1]["dtype"] == "int64"
    assert io_inputs_match(built, ref)
    broken = json.loads(json.dumps(built))
    broken["inputs"][0]["dims"] = [1, 60, 80]
    assert not io_inputs_match(broken, ref)


def test_build_metadata_json_dumps_with_type_objects():
    """Export metadata.json must survive Python type objects (k2 stub / NodeArg)."""
    with_cuda = type("with_cuda", (), {})
    meta = build_metadata(
        run="interp-multi-a0.5",
        epoch=1,
        avg=1,
        chunk=24,
        left=256,
        icefall_sha="3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1",
        k2="cpu-stub",
        param_count=64_684_786,
        fp32_bytes=259_593_848,
        int8_bytes=69_245_985,
        T=61,
        hop=48,
        io_diff=[],
        init_meta={
            "init_pt": "/vol/reference/zipformer_p_arabic_v3.1.pt",
            "ft_pt": Path("/vol/exp/ft-multi/epoch-1.pt"),
            "alpha": 0.5,
            "n_tensors": 829,
            "icefall_pt": Path("/vol/exp/interp-multi-a0.5/epoch-1.pt"),
            "init_space": "reference",
            "ft_state_key": "model",
            "ft_top_keys": ["epoch", "model", "model_avg", "optimizer"],
            "k2-with-cuda": with_cuda,
            "node_type": np.float32,
        },
    )
    dumped = json.dumps(meta)
    loaded = json.loads(dumped)
    assert loaded["run"] == "interp-multi-a0.5"
    assert loaded["T"] == 61
    assert loaded["hop"] == 48
    assert loaded["init_meta"]["init_space"] == "reference"
    assert loaded["init_meta"]["ft_state_key"] == "model"
    assert loaded["init_meta"]["ft_top_keys"][0] == "epoch"
    assert loaded["init_meta"]["ft_pt"] == "/vol/exp/ft-multi/epoch-1.pt"
    assert isinstance(loaded["init_meta"]["k2-with-cuda"], str)
    assert "with_cuda" in loaded["init_meta"]["k2-with-cuda"]
    assert isinstance(loaded["init_meta"]["node_type"], str)
    assert json_plain(with_cuda) == str(with_cuda)

    node = SimpleNamespace(name="x", shape=[1, 61, 80], type=np.float32)
    io = io_json_from_session(
        [node],
        [SimpleNamespace(name="log_probs", shape=[1, 24, 251], type=float)],
        model="model.onnx",
        T=61,
        hop=48,
    )
    json.dumps(io)
    assert io["inputs"][0]["dtype"] == "float32"


def test_arch_flags_reference_cnn_kernels():
    flags = list(ARCH_FLAGS)
    i = flags.index("--cnn-module-kernel")
    assert flags[i + 1] == "31,31,15,15,15,31"
    j = flags.index("--pos-dim")
    assert flags[j + 1] == "192"


def test_resolve_train_sources_full_run_fails_loud(tmp_path: Path):
    (tmp_path / "retasy_cuts_fbank.jsonl.gz").write_bytes(b"")
    requested = "everyayah,qua,iqra,retasy,tlog"
    assert missing_fbank_sources(requested, tmp_path) == [
        "everyayah",
        "qua",
        "iqra",
        "tlog",
    ]
    with pytest.raises(FileNotFoundError, match=r"missing staged fbank cuts") as exc:
        resolve_train_sources(requested, tmp_path)
    msg = str(exc.value)
    assert "everyayah" in msg and "tlog" in msg
    assert "retasy" not in missing_fbank_sources(requested, tmp_path)
    assert "dropping unstaged" not in msg

    for src in requested.split(","):
        (tmp_path / f"{src}_cuts_fbank.jsonl.gz").write_bytes(b"")
    got, use_syn = resolve_train_sources(requested, tmp_path)
    assert got == ["everyayah", "qua", "iqra", "retasy", "tlog"]
    assert use_syn is False
    still_real, still_syn = resolve_train_sources(requested, tmp_path, smoke=True)
    assert still_real == got and still_syn is False

    smoke_src, smoke_syn = resolve_train_sources(
        requested, tmp_path / "empty", smoke=True
    )
    assert smoke_src == ["synthetic"] and smoke_syn is True
    syn_src, syn_flag = resolve_train_sources(requested, tmp_path / "empty", synthetic=True)
    assert syn_src == ["synthetic"] and syn_flag is True

    train_py = (ROOT / "scripts" / "train_zipformer_ctc_modal.py").read_text()
    assert "resolve_train_sources" in train_py
    assert "dropping unstaged sources" not in train_py
    assert "falling back to --synthetic" not in train_py


def test_parse_export_interp_and_alpha_endpoints():
    init_pt, ft_pt, alpha, key = parse_export_interp(
        "/vol/reference/zipformer_p_arabic_v3.1.pt:/vol/exp/ft-v31/epoch-5.pt:0.25"
    )
    assert init_pt.endswith("zipformer_p_arabic_v3.1.pt")
    assert ft_pt.endswith("epoch-5.pt")
    assert alpha == 0.25
    assert key == "model"
    *_, key_avg = parse_export_interp(
        "/vol/reference/zipformer_p_arabic_v3.1.pt:/vol/exp/ft-v31/epoch-5.pt:0.5:model_avg"
    )
    assert key_avg == "model_avg"
    with pytest.raises(ValueError, match="INIT_PT:FT_PT:ALPHA"):
        parse_export_interp("only-one-path")
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        parse_export_interp("/a.pt:/b.pt:1.5")
    blob = {"model": {"a": 1}, "model_avg": {"a": 2}, "epoch": 5}
    assert select_checkpoint_state(blob, "model") == {"a": 1}
    assert select_checkpoint_state(blob, "model_avg") == {"a": 2}
    with pytest.raises(KeyError, match="top-level keys"):
        select_checkpoint_state({"model": {"a": 1}}, "model_avg")


def test_interpolate_alpha_endpoints_and_ctc_permute_before_blend():
    rng = np.random.default_rng(7)
    hidden = 4
    init_w = rng.standard_normal((VOCAB_SIZE, hidden)).astype(np.float32)
    init_b = rng.standard_normal(VOCAB_SIZE).astype(np.float32)
    init_w[250] = 10.0
    init_b[250] = 7.0
    init_w[0] = 1.0
    init_b[0] = 2.0
    ft_w = rng.standard_normal((VOCAB_SIZE, hidden)).astype(np.float32)
    ft_b = rng.standard_normal(VOCAB_SIZE).astype(np.float32)
    ft_w[0] = 20.0
    ft_b[0] = 14.0
    ft_w[1] = 3.0
    ft_b[1] = 4.0
    encoder_init = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    encoder_ft = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    init_sd = {
        "sub.conv.weight": encoder_init,
        "ctc_head.weight": init_w,
        "ctc_head.bias": init_b,
        "decoder.embedding.weight": np.ones(3, dtype=np.float32),
    }
    ft_sd = {
        "encoder_embed.conv.weight": encoder_ft,
        CTC_ICEFALL_WEIGHT_KEY: ft_w,
        CTC_ICEFALL_BIAS_KEY: ft_b,
    }

    ice_w, ice_b = inverse_permute_ctc_head(init_w, init_b)
    np.testing.assert_array_equal(ice_w[0], 10.0)
    assert ice_b[0] == 7.0
    np.testing.assert_array_equal(ice_w[1], 1.0)
    assert ice_b[1] == 2.0

    out0 = interpolate_state_dicts(init_sd, ft_sd, 0.0)
    np.testing.assert_allclose(out0["encoder_embed.conv.weight"], encoder_init)
    np.testing.assert_allclose(out0[CTC_ICEFALL_WEIGHT_KEY], ice_w)
    np.testing.assert_allclose(out0[CTC_ICEFALL_BIAS_KEY], ice_b)
    assert "decoder.embedding.weight" not in out0

    out1 = interpolate_state_dicts(init_sd, ft_sd, 1.0)
    np.testing.assert_allclose(out1["encoder_embed.conv.weight"], encoder_ft)
    np.testing.assert_allclose(out1[CTC_ICEFALL_WEIGHT_KEY], ft_w)
    np.testing.assert_allclose(out1[CTC_ICEFALL_BIAS_KEY], ft_b)

    out05 = interpolate_state_dicts(init_sd, ft_sd, 0.5)
    np.testing.assert_allclose(
        out05["encoder_embed.conv.weight"], 0.5 * encoder_init + 0.5 * encoder_ft
    )
    # Blend in icefall order: blank row 0 = 0.5 * ft[0] + 0.5 * init_ref[250]
    np.testing.assert_allclose(out05[CTC_ICEFALL_WEIGHT_KEY][0], 0.5 * 20.0 + 0.5 * 10.0)
    np.testing.assert_allclose(out05[CTC_ICEFALL_BIAS_KEY][0], 0.5 * 14.0 + 0.5 * 7.0)
    np.testing.assert_allclose(out05[CTC_ICEFALL_WEIGHT_KEY][1], 0.5 * 3.0 + 0.5 * 1.0)
    np.testing.assert_allclose(out05[CTC_ICEFALL_BIAS_KEY][1], 0.5 * 4.0 + 0.5 * 2.0)
    # Wrong order (blend then permute) would mix ref row 0 with icefall row 0.
    wrong_w = 0.5 * init_w + 0.5 * ft_w
    assert not np.allclose(out05[CTC_ICEFALL_WEIGHT_KEY][0], wrong_w[0])

    with pytest.raises(KeyError, match="key mismatch"):
        interpolate_state_dicts(
            init_sd,
            {**ft_sd, "extra.weight": np.ones(2, dtype=np.float32)},
            0.5,
        )

    # icefall-space init: do NOT inverse-permute again (alpha=0 identity).
    init_ice = {
        "encoder_embed.conv.weight": encoder_init,
        CTC_ICEFALL_WEIGHT_KEY: ice_w,
        CTC_ICEFALL_BIAS_KEY: ice_b,
    }
    assert detect_state_space(init_sd) == "reference"
    assert detect_state_space(init_ice) == "icefall"
    out_ice0 = interpolate_state_dicts(init_ice, ft_sd, 0.0)
    np.testing.assert_allclose(out_ice0["encoder_embed.conv.weight"], encoder_init)
    np.testing.assert_allclose(out_ice0[CTC_ICEFALL_WEIGHT_KEY], ice_w)
    np.testing.assert_allclose(out_ice0[CTC_ICEFALL_BIAS_KEY], ice_b)
    twice = inverse_permute_ctc_head(ice_w, ice_b)
    assert not np.allclose(out_ice0[CTC_ICEFALL_BIAS_KEY], twice[1])

    with pytest.raises(ValueError, match="ambiguous"):
        detect_state_space([*init_sd, *init_ice])
    with pytest.raises(ValueError, match="cannot detect"):
        detect_state_space(["encoder_embed.conv.weight"])

    train_py = (ROOT / "scripts" / "train_zipformer_ctc_modal.py").read_text()
    assert "export_interp" in train_py
    assert "parse_export_interp" in train_py
    assert "ft_state_key" in train_py
    assert "icefall_state_from_reference" in train_py
    assert "init_is_reference" not in train_py


def test_default_train_sources_exclude_qurantts():
    parts = DEFAULT_TRAIN_SOURCES.split(",")
    assert "qurantts" not in parts
    assert parts == ["everyayah_rx", "qua_rx", "iqra_rx", "retasy_rx", "tlog_clean_v3"]
    train_py = (ROOT / "scripts" / "train_zipformer_ctc_modal.py").read_text()
    data_py = (ROOT / "scripts" / "zipformer_asr_datamodule.py").read_text()
    assert train_py.count("sources: str = DEFAULT_TRAIN_SOURCES") == 2
    assert 'default="everyayah_rx,qua_rx,iqra_rx,retasy_rx,tlog_clean_v3"' in data_py
    assert "--sources everyayah,qua,qurantts" not in train_py
