"""Small synthetic checks; real acoustic experiments run only on Modal."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_articulation import (occupancy, safe_intervals, wsola, duration_view,
    global_view, speech_islands, admitted_none, speaker_partition)


def enumerated(p, target):
    states = [0]
    for v in target:
        states += [v, 0]
    mass = np.zeros((len(p), len(states)))
    enter, leave = np.zeros((len(p), len(target))), np.zeros((len(p), len(target)))
    z = 0.
    for path in itertools.product(range(len(states)), repeat=len(p)):
        if path[0] not in (0, 1) or path[-1] not in (len(states) - 2, len(states) - 1):
            continue
        if any(not (b == a or b == a + 1 or
                    (b == a + 2 and states[b] and states[a] != states[b]))
               for a, b in zip(path, path[1:])):
            continue
        weight = np.prod([p[t, states[s]] for t, s in enumerate(path)])
        z += weight
        for t, s in enumerate(path):
            mass[t, s] += weight
        for i in range(len(target)):
            times = [t for t, s in enumerate(path) if s == 2 * i + 1]
            enter[times[0], i] += weight
            leave[times[-1], i] += weight
    return mass / z, enter / z, leave / z, np.log(z)


@pytest.mark.parametrize("target", [[1], [1, 2], [1, 1]])
def test_ctc_marginals_match_all_paths_including_repeated_occurrences(target):
    p = np.random.default_rng(3).uniform(.1, 1., (4, 3))
    p /= p.sum(1, keepdims=True)
    got = occupancy(np.log(p), target)
    want = enumerated(p, target)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(got[0].sum(1), 1.)
    np.testing.assert_allclose(got[1].sum(0), 1.)
    np.testing.assert_allclose(got[2].sum(0), 1.)


def test_impossible_or_invalid_alignment_fails():
    with pytest.raises(ValueError, match="Impossible"):
        occupancy(np.log(np.full((2, 3), 1 / 3)), [1, 1])
    for bad in ([0], [3], [], [1.5]):
        with pytest.raises(ValueError):
            occupancy(np.log(np.full((4, 3), 1 / 3)), bad)
    with pytest.raises(ValueError):
        occupancy(np.full((4, 3), np.nan), [1])


def strong_phone_path():
    ids = [0] * 2 + [1] * 5 + [2] * 5 + [0] + [3] * 5 + [0] * 7
    p = np.full((len(ids), 4), 1e-8)
    p[np.arange(len(ids)), ids] = 1 - 3e-8
    return np.log(p), len(ids) * 640


def test_cores_skip_vowels_and_preserve_uncertain_boundary_guards():
    lp, samples = strong_phone_path()
    spans, z = safe_intervals(lp, "بات", [1, 2, 3], samples)
    assert [s["occurrence"] for s in spans] == [0, 2]
    assert [(s["start"], s["end"]) for s in spans] == [(1440, 4320), (8480, 11360)]
    assert np.isfinite(z)
    spans, _ = safe_intervals(np.log(np.full((20, 3), 1 / 3)), "بت", [1, 2], 12800)
    assert spans == []  # Diffuse onset/offset distributions do not license a cut.


def test_sustained_nasal_and_madd_are_never_consonant_cores():
    lp, samples = strong_phone_path()
    spans, _ = safe_intervals(lp, "ںۦۥ", [1, 2, 3], samples)
    assert spans == []


@pytest.mark.parametrize("length", [640, 1600, 3200, 4800])
def test_identity_uses_overlap_add_and_is_sample_exact(length):
    wave = np.random.default_rng(4).normal(0, .1, length).astype(np.float32)
    np.testing.assert_array_equal(wsola(wave, length), wave)


@pytest.mark.parametrize("factor", [.75, 1.5])
def test_duration_change_preserves_voiced_pitch_instead_of_resampling(factor):
    wave = np.sin(2 * np.pi * 220 * np.arange(3200) / 16000).astype(np.float32)
    got = wsola(wave, int(len(wave) * factor))
    assert len(got) == int(len(wave) * factor) and np.isfinite(got).all()
    pitch = np.argmax(abs(np.fft.rfft(got))) * 16000 / len(got)
    assert abs(pitch - 220) < 6
    assert got[0] == wave[0] and got[-1] == wave[-1]


def test_only_consonant_cores_change_protected_audio_is_copied_in_order():
    wave = np.random.default_rng(6).normal(0, .1, 8000).astype(np.float32)
    spans = [dict(start=1000, end=2600, phone="ب", occurrence=0),
             dict(start=5000, end=6600, phone="ت", occurrence=2)]
    got, mapping = duration_view(wave, spans, [1.25, .75])
    assert len(got) == len(wave)
    np.testing.assert_array_equal(got[:1000], wave[:1000])
    np.testing.assert_array_equal(got[3000:5400], wave[2600:5000])  # Entire neighbouring vowel/pause.
    np.testing.assert_array_equal(got[6600:], wave[6600:])
    assert [(m["outputStart"], m["outputEnd"]) for m in mapping] == [(1000, 3000), (5400, 6600)]
    assert [m["occurrence"] for m in mapping] == [0, 2]


def test_global_control_matches_duration_and_copies_real_pauses():
    wave = np.random.default_rng(7).normal(0, .1, 10000).astype(np.float32)
    islands = [dict(start=1000, end=4000), dict(start=6000, end=9000)]
    cores = [dict(start=1500, end=2500), dict(start=6500, end=7500)]
    factors = [1.25, .8]
    relative, _ = duration_view(wave, cores, factors)
    global_audio, mapping = global_view(wave, islands, cores, factors)
    assert len(global_audio) == len(relative) == 10050
    np.testing.assert_array_equal(global_audio[:1000], wave[:1000])
    np.testing.assert_array_equal(global_audio[4250:6250], wave[4000:6000])
    np.testing.assert_array_equal(global_audio[9050:], wave[9000:])
    assert mapping[0]["outputEnd"] - mapping[0]["outputStart"] == 3250
    assert mapping[1]["outputEnd"] - mapping[1]["outputStart"] == 2800


def test_blank_runs_are_protected_without_discarding_weak_speech():
    lp = np.log(np.tile([.5, .5], (25, 1)))
    lp[5:10] = np.log([.99, .01])
    lp[12:16] = np.log([.99, .01])  # Too brief and may be within-word blank.
    lp[20:] = np.log([.99, .01])
    assert speech_islands(lp, 25 * 640) == [dict(start=0, end=3200), dict(start=6400, end=12800)]


def test_bad_or_overlapping_duration_intervals_fail():
    wave = np.ones(3200, dtype=np.float32)
    for spans, factors in [([dict(start=0, end=1000), dict(start=900, end=1900)], [1, 1]),
                           ([dict(start=0, end=1000)], [np.nan]),
                           ([dict(start=0, end=1000)], [.1]),
                           ([dict(start=0, end=1000)], [])]:
        with pytest.raises(ValueError):
            duration_view(wave, spans, factors)
    with pytest.raises(ValueError, match="speech island"):
        global_view(wave, [dict(start=0, end=1000)], [dict(start=900, end=1900)], [1.2])


def none_row(i, speaker):
    return dict(id=str(i), speaker=speaker, split="train", kind="none", scored=True,
                events=[], non_errors=[], unsupported=[], wav="take.wav", ayahs=["1:1"])


def test_only_clean_none_takes_from_consistently_training_speakers_enter():
    clean = none_row(1, "a")
    assert admitted_none([clean]) == [clean]
    for updates in [dict(kind="repeat"), dict(scored=False), dict(events=[{}]),
                    dict(unsupported=[{}]), dict(non_errors=[{}]), dict(excluded=True)]:
        assert admitted_none([{**clean, **updates}]) == []
    assert admitted_none([clean, {**none_row(2, "a"), "split": "test"}]) == []
    assert admitted_none([clean], ["a"]) == []
    with pytest.raises(ValueError, match="Duplicate"):
        admitted_none([clean, clean])


def test_speaker_partition_is_order_invariant_and_groups_all_takes():
    speakers = list("abcdefghij")
    groups = speaker_partition(speakers)
    assert groups == speaker_partition(speakers[::-1] + ["a"])
    assert sum(v == "selection" for v in groups.values()) == 2
    with pytest.raises(ValueError):
        speaker_partition(["a", "b"])


def test_training_cli_distinguishes_help_labels_from_usage_help():
    from student_articulation_train import parser
    args = parser().parse_args(["prepare", "--model", "model", "--help-labels", "labels",
        "--transcribed", "transcripts", "--exclusions", "excluded", "--augmentation", "banks", "--out", "cache"])
    assert args.command == "prepare" and args.help == "labels"
    with pytest.raises(SystemExit) as result:
        parser().parse_args(["prepare", "--help"])
    assert result.value.code == 0


def test_selection_is_read_only_and_weights_speakers_equally():
    import torch
    from student_articulation_train import phone_head, validate, recording_loss
    head = phone_head(2, 3)
    v = dict(features=torch.randn(8, 2), shipped=torch.randn(8, 3).log_softmax(-1),
             target=torch.tensor([1, 2]))
    w = {**v, "target": torch.tensor([2, 1, 2])}
    a, b = dict(speaker="a", common=[v] * 3), dict(speaker="b", common=[w] * 3)
    before = {k: p.clone() for k, p in head.state_dict().items()}
    assert validate(head, [a, b]) == validate(head, [a] * 9 + [b])
    assert all(torch.equal(p, before[k]) for k, p in head.state_dict().items())
    assert all(p.grad is None for p in head.parameters())
    row = dict(common=[v] * 3, arms={k: [v] * 3 for k in ("identity", "global", "relative")})
    losses = [recording_loss(head, row, k) for k in row["arms"]]
    assert all(torch.equal(losses[0], value) for value in losses)


def test_verified_authorized_enrollment_excludes_speakers_across_both_corpora(tmp_path):
    import json
    from student_articulation_train import recordings
    from student_head_export import digest
    help_audio, other_audio = tmp_path / "help.wav", tmp_path / "other.wav"
    help_audio.write_bytes(b"help recording")
    other_audio.write_bytes(b"authorized recording")
    help_labels, transcribed = tmp_path / "help.jsonl", tmp_path / "transcribed.jsonl"
    help_labels.write_text(json.dumps({**none_row(1, "help"), "wav": "help.wav"}) + "\n")
    extra = dict(speaker="other", split="train", authorized=True, verified=True,
                 phones="ب", wav="other.wav", sha256=digest(other_audio))
    transcribed.write_text(json.dumps(extra) + "\n")
    ref = dict(ayahs=[dict(s=1, a=1, p="ب")])
    rows, excluded = recordings(help_labels, transcribed, [], ref)
    assert len(rows) == 2 and excluded == []
    for change in [dict(verified=False), dict(authorized=False), dict(sha256="0" * 64)]:
        transcribed.write_text(json.dumps({**extra, **change}) + "\n")
        with pytest.raises(ValueError):
            recordings(help_labels, transcribed, [], ref)
    transcribed.write_text(json.dumps(extra) + "\n" + json.dumps(dict(speaker="help", split="test")))
    with pytest.raises(ValueError, match="Both authorized"):
        recordings(help_labels, transcribed, [], ref)
