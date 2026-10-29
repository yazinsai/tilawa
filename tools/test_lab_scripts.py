"""uv run --python cpython-3.12-macos-aarch64-none --with quran-transcript==0.6.4 --with rapidfuzz==3.14.1 --with pytest pytest tools/test_lab_scripts.py"""

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_scripts as T  # noqa: E402

ASSET = json.loads((T.ROOT / "packages/core/assets/ref-v2.json").read_text())
Q = T.Quran(ASSET)
BATCH = T.batch(ASSET)
TOKENS = set(ASSET["inventory"]["tokens"])


def ayah_index(s, at):
    return next(i for i, row in enumerate(r for r in s["ayahs"] if not r.get("basmala")) if (row["surah"], row["ayah"]) == (at["surah"], at["ayah"]))


def test_deterministic_for_a_seed():
    assert T.batch(ASSET) == BATCH
    assert T.batch(ASSET, seed=T.SEED + 1)["scripts"] != BATCH["scripts"]


def test_batch_shape_and_length():
    ss = BATCH["scripts"]
    assert len(ss) == 10 and len({s["id"] for s in ss}) == 10
    assert all(T.MINUTES[0] * 60 <= s["estimatedS"] <= T.MINUTES[1] * 60 for s in ss)
    assert 30 <= BATCH["summary"]["estimatedMinutes"] <= 45
    surahs = [x["surah"] for s in ss for x in s["ayahs"] if not x.get("basmala")]
    by_script = [{x["surah"] for x in s["ayahs"]} for s in ss]
    assert sum(len(b) for b in by_script) == len(set().union(*by_script)), "a surah is used twice"
    assert {s["tag"] for s in ss} == {"juz-amma", "juz-tabarak", "long"} and ss[0]["tag"] == "juz-amma"
    assert sum(s["clean"] for s in ss) == 2 and all(not s["plants"] for s in ss if s["clean"])
    assert all(s["counts"].get("self_correction", 0) >= 1 for s in ss if not s["clean"])
    assert len(surahs) == BATCH["summary"]["ayahs"]


def test_plant_spacing_and_cap():
    for s in BATCH["scripts"]:
        idx = [ayah_index(s, p["location"]) for p in s["plants"]]
        assert all(i >= T.FIRST_PLANT[0] for i in idx)
        assert all(b - a >= 2 for a, b in zip(idx, idx[1:]))
        assert all(i < s["ayahCount"] - 1 for i in idx)
        g = [Q.ref.idx[(p["location"]["surah"], p["location"]["ayah"], p["location"]["word"])] for p in s["plants"]]
        assert all(b - a >= T.GAP[0] for a, b in zip(g, g[1:]))
        mistakes = [p for p in s["plants"] if p["class"] != "self_correction"]
        assert sum(p["class"] == "added_word" for p in mistakes) <= math.ceil(T.ADDED_MAX * len(mistakes))


def test_phonetize_run_reproduces_the_reference():
    for s, a in [(1, 2), (2, 2), (77, 15), (104, 1), (112, 1)]:
        ids = Q.words_of(s, a)
        assert T.phonetize_run([Q.ref.words[g]["text"] for g in ids], s) == [Q.ref.words[g]["cont"] for g in ids]


def test_planted_ayahs_said_in_context():
    s = next(s for s in BATCH["scripts"] if s["id"] == "rec-v1-07")
    for p in s["plants"]:
        if p["class"] in ("wrong_word", "added_word"):
            x = next(x for x in s["said"] if x.get("plant") == p["id"])
            assert x["phonemes"] == p["saidPhonemes"] and set(x["phonemes"]) <= TOKENS
    sc = next(p for s in BATCH["scripts"] for p in s["plants"] if p["class"] == "self_correction")
    s = next(s for s in BATCH["scripts"] if sc in s["plants"])
    slip = next(x for x in s["said"] if x.get("plant") == sc["id"] and x["role"] == "slip")
    assert slip["phonemes"] == sc["saidPhonemes"]


def test_said_matches_reference_outside_plants():
    for s in BATCH["scripts"]:
        for row in s["ayahs"]:
            if row.get("basmala") or row["plants"]:
                continue
            ph = "".join(x["phonemes"] for x in s["said"] if x["at"] and (x["at"]["surah"], x["at"]["ayah"]) == (row["surah"], row["ayah"]))
            assert ph == Q.p[(row["surah"], row["ayah"])]
    first = BATCH["scripts"][0]
    assert [x["role"] for x in first["said"][:2]] == ["istiadha", "basmala"] and first["ayahs"][0]["basmala"]
    assert first["said"][-1]["role"] == "closing" and set(first["said"][0]["phonemes"]) <= TOKENS


def test_plants_are_said_as_scripted():
    for s in BATCH["scripts"]:
        for p in s["plants"]:
            mine = [x for x in s["said"] if x.get("plant") == p["id"]]
            if p["class"] == "omitted_word":
                assert not mine and not any(x["at"] == p["location"] for x in s["said"])
                assert p["instruction"] == f"Leave out «{p['expected']}»."
            elif p["class"] == "wrong_word":
                (x,) = mine
                assert x["role"] == "wrong" and x["at"] == p["location"] and x["text"] == p["said"] != p["expected"]
                g = Q.ref.idx[(p["location"]["surah"], p["location"]["ayah"], p["location"]["word"])]
                assert Q.ref.words[Q.ref.idx[(p["saidFrom"]["surah"], p["saidFrom"]["ayah"], p["saidFrom"]["word"])]]["skel"] not in {Q.sk(g, d) for d in (-1, 0, 1)}
                assert p["said"] in p["instruction"] and p["expected"] in p["instruction"]
            elif p["class"] == "added_word":
                (x,) = mine
                assert x["role"] == "added" and x["at"] is None and p["expected"] == ""
                k = s["said"].index(x)
                assert s["said"][k + 1]["at"] == p["location"]
            else:
                roles = [x["role"] for x in mine]
                assert roles[0] == "slip" and set(roles[1:]) == {"retake"} and 2 <= len(roles) <= 3
                assert mine[-1]["at"] == p["location"] and mine[-1]["text"] == p["expected"]
                assert [x["at"] for x in mine[1:]] == p["retake"]


def test_labels_known_by_construction():
    for s in BATCH["scripts"]:
        ev, ne = s["labels"]["events"], s["labels"]["nonErrors"]
        assert len(ev) == sum(p["class"] != "self_correction" for p in s["plants"])
        for e in ev:
            assert e["errorClass"] in ("wrong_word", "added_word", "omitted_word") and e["state"] == "confirmed"
            if e["errorClass"] != "omitted_word":
                assert e["said"] and set(e["said"]) <= TOKENS
            assert isinstance(e["forms_other_ayah"], bool)
        assert len(ne) == sum(len(p["retake"]) for p in s["plants"] if p["class"] == "self_correction")


def test_sha_tracks_content():
    s = BATCH["scripts"][0]
    body = {k: v for k, v in s.items() if k != "sha"}
    assert T.digest(body) == s["sha"]
    assert T.digest({**body, "title": "x"}) != s["sha"]


def test_twin_substitute_first():
    found = 0
    for g in sorted(Q.twins)[:400]:
        subs = T.substitutes(Q, g, random.Random(1))
        if any(c["kind"] == "twin" for c in subs):
            assert subs[0]["kind"] == "twin"
            found += 1
    assert found > 10


def test_lookalikes_change_one_letter():
    for d, f, sk, j in T.lookalike_words(Q, "قالوا"):
        assert d == 1 and (len(sk) == 5 and sum(a != b for a, b in zip(sk, "قالوا")) == 1 or sk[1:] == "قالوا")


def test_custom_pool():
    out = T.batch(ASSET, n=1, pool=[{"tag": "x", "surahs": [36], "weight": 1}], clean=0)
    (s,) = out["scripts"]
    assert s["tag"] == "x" and s["from"]["surah"] == s["to"]["surah"] == 36 and s["plants"]


def test_text_lists_every_instruction():
    s = next(s for s in BATCH["scripts"] if not s["clean"])
    t = T.text(s)
    assert all(p["instruction"] in t for p in s["plants"])
