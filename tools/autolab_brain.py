"""Autolab orchestrator calls (.scratch/autolab/spec.md decisions 1, 8, 9, 18; ticket 06). Each call is stateless: a fixed
prompt, a context pack built by code, and a JSON answer checked against a schema. A malformed answer is retried once
with the error; a second failure raises BrainError and the supervisor marks the work needs-human.

    plan(cfg, pack, slots, st)          Opus: one card per free (component, tier) slot
    judge(cfg, card, pack)              Grok:  is this a restatement of a tried mechanism?
    review_diff(cfg, card, diff)        Grok:  does the diff special-case the benchmark or weaken checks?
    review(cfg, exp, outcome, pack)     Opus: the journal entry, and a follow-up card or why the line is dead
    strategy(cfg, pack, frozen, slots, st)  Opus: the "why stuck" memo and bets

Models come from cfg["models"] by role (planner, judge, diffReviewer, reviewer, strategist). `ask` defaults to the
Cursor SDK one-shot call (tools/autolab_cursor.py prompt_model); tests pass their own.
"""

import json
import re

import autolab_portfolio as P
from lab_ledger import COMPONENTS, ID

TIERS = ("exploit", "explore", "bet")
MAX_DIFF = 60_000


class BrainError(Exception):
    pass


class Unavailable(BrainError):
    """The model didn't answer (at capacity, timed out, network): try again on a later tick; not the experiment's fault."""


ABOUT = """Tilawa is an offline streaming Quran recitation engine. A reciter recites freely; the engine must warn on real
mistakes (wrong, omitted or added words, skipped passages, mutashabihat swaps) and stay quiet otherwise. Three components:
- listener: the acoustic model (CTC phoneme posteriors) and its decoding (packages/sdk, modal/student_*.py);
- tracker: finds where in the Quran the reciter is (packages/core/src/tracker.ts);
- checker: compares what was heard with the text at that place and decides warnings (packages/core/src/checker.ts).
The lab scores the engine on a practice set of 100 spliced sessions with planted mistakes, phone-degraded audio. The score
is wrong calls: missed mistakes plus false warnings, lower is better. Attribution charges each wrong call to the component
that lost it, so listener + tracker + checker = overall. Scoring is deterministic. A change of 2 or fewer is noise.
An incident key looks like `<session>/phone/mistake:<class>/<surah:ayah:word>/<seconds>` (a missed mistake) or a false
warning's key. A cluster groups incidents by charged component, error class and kind.

The fold gate (code, not you) keeps a change only if overall improves by 3 or more, its component gets no more than 2
worse, the gain survives leaving out the reader who gained most, no new false warnings appear on fully clean sessions,
studio audio gets no more than 2 worse, and incidents lost are at most 30% of those won (any loss needs a net gain of 5).
A diff that only changes numeric constants is "tuning" and needs a gain of 5. Benchmark-specific code (session ids,
reader ids, verse references) and edits to the scorer or test data are rejected outright."""

CARD_SCHEMA = """A card is a JSON object:
{"id": "<component>-<short-slug>", lowercase letters, digits and dashes, at most 40 characters, new;
 "component": "listener" | "tracker" | "checker";
 "tier": "exploit" | "explore" | "bet";
 "hypothesis": one line: what is wrong now and what change fixes it, with the predicted counts;
 "eli5": one plain sentence (at most 160 characters) a smart 12-year-old would follow: what goes wrong for the reciter
   and what the change does about it; no jargon (no CTC, posterior, checker, tracker, listener, alignment, threshold);
 "mechanism": 2-5 sentences: the causal story, which code path, why the targeted incidents fail, why the change fixes them
   without new losses, and how it differs from any tried mechanism on the same cluster;
 "cluster": one cluster key from the context;
 "targets": [incident keys from that cluster that this change should win];
 "expected": {"component": <int, negative = fewer wrong calls>, "overall": <int>};
 "killCriterion": one line: a check on the targets that shows the premise is false (for example "fewer than 3 of the
   targets have <the property the mechanism relies on>"). Never a collateral clause ("or any clean seam / pause / session
   gains ..."): the fold gate already prices losses against wins, and a builder that stops on one counterexample never
   learns whether the trade would have folded;
 "reason": required only when the cluster has 2 or more failed tries: why this attempt is different;
 "gpuUsd": bets only: the Modal GPU budget in dollars; "earlyCheck": bets only: a cheap check and its kill threshold}
Tiers: exploit = a bounded fix to one incident cluster (hours of work); explore = a new mechanism (at most a day);
bet = retraining, new training data, or a decoder or architecture change (days of GPU)."""

PLAN = """{about}

You are the research lead. Write exactly one experiment card for each free slot below, in order. Each card must take its
slot's component and tier.

How to choose:
- Work from the incidents, not from knobs. Pick a cluster with many incidents for the slot's component, read what has
  been tried on it (history and journal), and propose a mechanism that explains why those incidents fail.
- Do not restate a tried mechanism. A different threshold on the same idea is a restatement. If every obvious idea on a
  cluster has failed, change the level of attack (data, model, representation) rather than the constant.
- Prefer changes whose gain would generalize to unseen reciters and passages. Never special-case sessions or verses.
- Predictions are counts on the practice set; be honest, small gains are fine. The context's "calibration" lists past
  predictions against what was measured: if they ran optimistic, predict lower, and prefer mechanisms that reach more
  incidents over ones that need every targeted case to flip. A change measured below the gate's bar but with nothing
  lost is a near miss worth building on (combine it with a related fix rather than restating it).
- Bets should be the kind of change the history shows small tweaks can't reach.

{schema}

Context (JSON):
{pack}

Free slots, in order: {slots}

Answer with one ```json block: {{"cards": [<card>, ...]}}"""

JUDGE = """{about}

You are a skeptical reviewer guarding against research going in circles. Decide whether this proposed experiment is
genuinely new, or a restatement of a mechanism already tried (folded, discarded or superseded) on the same incidents. A
restatement includes: the same idea with a different constant; the same code path patched the same way; a narrower or
broader version of a failed rule without a new reason it would work. An experiment with "measured": false never ran (it
was blocked, or timed out before a result): it tested nothing, so a card that finally runs that idea is not a restatement
of it.

Proposed card:
{card}

History of tried experiments (JSON):
{history}

Answer with one ```json block: {{"accept": true | false, "restates": [<experiment ids>], "why": "<one or two sentences>"}}"""

DIFF_REVIEW = """{about}

You review an experiment's diff before it is scored. Reject it if it special-cases the benchmark in any way the literal
checks might miss: logic keyed on specific sessions, readers, verses, positions, durations or audio quirks of the test
set; edits that weaken or skip checks only on paths the test set exercises; changes to tests that make them pass rather
than test; anything touching scoring, test data or lab tooling. Ordinary engine changes, even big ones, are fine.

Card:
{card}

Diff (may be truncated):
{diff}

Answer with one ```json block: {{"ok": true | false, "why": "<one or two sentences>", "concerns": ["<file:line - concern>", ...]}}"""

REVIEW = """{about}

You keep the lab journal. An experiment just ended. Write its journal entry in plain words (what was tried, what happened,
what it teaches), then either propose one follow-up card or say why this line of work is dead.

Experiment (card, executor summary, outcome with gate checks and incident changes):
{exp}

Context (JSON):
{pack}

{schema}

Answer with one ```json block:
{{"journal": "<markdown, 3-10 sentences>", "followUp": <card or null>, "deadEnd": "<why, or null>"}}"""

STRATEGY = """{about}

The lab is stuck: {frozen}. Small experiments on these components stopped paying off. Write a short memo on why (look at
what was tried, what the remaining incidents have in common, and what the attempts assumed), then propose cards for the
free slots below that attack the problem at a different level. Prefer bets and explores.

{schema}

Context (JSON):
{pack}

Free slots, in order: {slots}

Answer with one ```json block: {{"memo": "<markdown>", "cards": [<card>, ...]}}"""


ELI5 = """Tilawa is an app that listens to someone reciting the Quran from memory and warns them when they make a mistake
(a wrong, missing or extra word, or a skipped passage), while staying quiet when they recite correctly. Engineers run
experiments to make it catch more mistakes and raise fewer false alarms.

For each experiment below, write one plain sentence (at most 160 characters) a smart 12-year-old would follow: what goes
wrong for the reciter today and what the change does about it. No jargon: don't say CTC, posterior, checker, tracker,
listener, alignment, threshold, attribution or nats.

Experiments (JSON, id -> hypothesis):
{items}

Answer with one ```json block: {{"<id>": "<sentence>", ...}} covering every id."""


def eli5(cfg: dict, items: dict[str, str], ask=None) -> dict[str, str]:
    """One-line plain explanations for experiments that predate the card's eli5 field."""
    def check(a):
        if not isinstance(a, dict) or set(a) != set(items) or not all(isinstance(v, str) and v.strip() and len(v) <= 200 for v in a.values()):
            raise ValueError("answer every id with one sentence of at most 160 characters")
        return {k: v.strip() for k, v in a.items()}
    return call(cfg, "planner", ELI5.format(items=dump(items)), check, ask) if items else {}


# Calls


TIMEOUTS = {"planner": 2700, "strategist": 2700, "reviewer": 1500}


def default_ask(cfg: dict, role: str, prompt: str, timeout_s: int | None = None) -> str:
    """One model call in a child process with a hard wall-clock limit: the SDK can block before its own timeout starts,
    so on expiry the whole process group (SDK bridge included) is killed. Writing a plan's cards takes Opus longer than
    a verdict takes Grok, so the limit is per role (config `brainTimeouts` overrides)."""
    timeout_s = timeout_s or {**TIMEOUTS, **cfg.get("brainTimeouts", {})}.get(role, 900)
    import os  # noqa: PLC0415
    import signal  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    p = subprocess.Popen([sys.executable, str(Path(__file__).with_name("autolab_cursor.py")), "prompt"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(json.dumps({"model": cfg["models"][role], "prompt": prompt, "timeout": timeout_s}), timeout=timeout_s + 60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        raise Unavailable(f"{role}: no answer within {timeout_s + 60} s") from None
    try:
        res = json.loads(out.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise Unavailable(f"{role}: {(err or out)[-300:]}") from None
    if "error" in res:
        raise Unavailable(f"{role}: {res['error'][:300]}")
    return res["text"]


FINAL = ("\n\nYou have no tools and cannot read files, run commands or look anything up: everything you need is above. "
         "Don't describe what you would check. Decide now, and answer with only the JSON block.")


def extract(text: str):
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return json.loads(blocks[-1])
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON in the answer (it began: {text.strip()[:120]!r})")
    return json.JSONDecoder().raw_decode(text[start:])[0]


FALLBACK = {"judge": "reviewer", "diffReviewer": "reviewer"}


def call(cfg: dict, role: str, prompt: str, check, ask=None):
    """Asks, parses and checks; one retry with the error appended. A role whose model doesn't answer is asked once more
    on its fallback role's model (Grok can think past any limit on a large diff)."""
    ask = ask or default_ask
    err = None
    for attempt in range(2):
        try:
            text = ask(cfg, role, prompt + FINAL if not err else f"{prompt}{FINAL}\n\nYour previous answer was rejected: {err}\nAnswer again with only the JSON block.")
        except Unavailable:
            if role in FALLBACK and cfg["models"].get(FALLBACK[role]) != cfg["models"].get(role):
                return call(cfg, FALLBACK[role], prompt, check, ask)
            raise
        except Exception as e:  # noqa: BLE001 - any failure to get an answer is an outage, not a bad answer
            raise Unavailable(f"{role}: {e}") from e
        try:
            return check(extract(text))
        except (ValueError, KeyError, TypeError) as e:
            err = str(e)
    raise BrainError(f"{role}: {err}")


def dump(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=1)


# Context


BLOCKED = re.compile(r"^restates |changes no engine code|committed no change|ran too long|failed twice|ran past|strips? posteriors")


def history(lab_ledgers: list[dict], st: dict) -> list[dict]:
    """Every experiment tried so far: the human lab's and autolab's ledgers, plus autolab cards still in flight."""
    out, seen = [], set()
    for led in lab_ledgers:
        for id, e in led.get("experiments", {}).items():
            if id in seen:
                continue
            seen.add(id)
            c, d = e["card"], e.get("decision") or {}
            out.append({"id": id, "component": c["component"], "hypothesis": c["hypothesis"], "expected": c["expected"],
                        "state": e["state"], "note": d.get("note", ""), "measured": bool(e.get("results")) or e["state"] in ("open", "folded", "superseded")})
    for id, e in st["experiments"].items():
        c = e["card"]
        r = e.get("results") or {}
        row = {"id": id, "component": c["component"], "tier": e["tier"], "hypothesis": c["hypothesis"],
               "mechanism": c.get("mechanism", ""), "cluster": c.get("cluster"), "state": e["state"], "note": e.get("note", ""), "created": e.get("created", ""),
               "measured": not (e["state"] in ("discarded", "abandoned") and BLOCKED.search(e.get("note", "")))}
        if id in seen:
            out = [{**x, **row} if x["id"] == id else x for x in out]
        else:
            seen.add(id)
            out.append(row)
    return out


def pack(st: dict, clusters: dict[str, list[str]], lab_ledgers: list[dict], journal: str, journal_chars: int = 12_000) -> dict:
    """What every orchestrator call sees: the main line's practice score, the clusters with their incidents and coverage,
    the history, and the journal's most recent entries. Practice only."""
    cov = st["coverage"]
    main = st.get("main") or {}
    return {"testSet": st["testSet"], "main": {"practice": main.get("practice"), "attribution": main.get("attribution")},
            "calibration": calibration(st),
            "clusters": [{"key": k, "incidents": keys, "state": cov.get(k, {}).get("state", "untouched"),
                          "tried": cov.get(k, {}).get("experiments", []), "failedTries": P.failed_tries(st, k)}
                         for k, keys in sorted(clusters.items(), key=lambda kv: -len(kv[1]))],
            "frozen": st["counters"]["frozen"], "history": history(lab_ledgers, st), "journal": journal[-journal_chars:]}


def calibration(st: dict) -> list[dict]:
    """Predicted against measured practice change for every autolab experiment that got a number (the branch score, or
    what its builder measured before killing it)."""
    out = []
    for e in st["experiments"].values():
        r = e.get("results") or {}
        got = ((r.get("branchRun") or {}).get("compare") or {}).get("overall", {}).get("change")
        if got is None:
            got = ((r.get("builderOutcome") or {}).get("measured") or {}).get("overall")
        if isinstance(got, int):
            out.append({"id": e["id"], "predicted": e["card"]["expected"]["overall"], "measured": got})
    return out


# Checks


def check_card(c: dict, ctx: dict, taken: set[str], slot: tuple[str, str] | None = None) -> dict:
    if not isinstance(c, dict):
        raise ValueError("a card must be an object")
    if not ID.match(str(c.get("id", ""))) or c["id"] in taken:
        raise ValueError(f"card id {c.get('id')!r}: new, lowercase letters, digits and dashes, up to 40")
    if c.get("component") not in COMPONENTS or c.get("tier") not in TIERS:
        raise ValueError(f"{c['id']}: component and tier must be valid")
    if slot and (c["component"], c["tier"]) != slot:
        raise ValueError(f"{c['id']}: must take its slot {slot[0]}/{slot[1]}")
    for k in ("hypothesis", "mechanism", "killCriterion"):
        if not str(c.get(k, "")).strip():
            raise ValueError(f"{c['id']}: {k} is required")
    if "\n" in c["hypothesis"]:
        raise ValueError(f"{c['id']}: hypothesis must be one line")
    eli5 = str(c.get("eli5", "")).strip()
    if not eli5 or "\n" in eli5 or len(eli5) > 200:
        raise ValueError(f"{c['id']}: eli5 must be one plain sentence of at most 160 characters")
    e = c.get("expected") or {}
    if not all(isinstance(e.get(k), int) and not isinstance(e.get(k), bool) for k in ("component", "overall")):
        raise ValueError(f"{c['id']}: expected.component and expected.overall must be whole counts")
    clusters = {x["key"]: x for x in ctx["clusters"]}
    if c.get("cluster") not in clusters:
        raise ValueError(f"{c['id']}: cluster {c.get('cluster')!r} is not one of the context's clusters")
    bad = [t for t in c.get("targets") or [] if t not in clusters[c["cluster"]]["incidents"]]
    if not c.get("targets") or bad:
        raise ValueError(f"{c['id']}: targets must be incident keys of cluster {c['cluster']}" + (f" (not: {bad[:3]})" if bad else ""))
    if clusters[c["cluster"]]["failedTries"] >= P.FAILED_TRIES and not str(c.get("reason", "")).strip():
        raise ValueError(f"{c['id']}: cluster {c['cluster']} has {clusters[c['cluster']]['failedTries']} failed tries; give a reason")
    if c["tier"] == "bet" and not isinstance(c.get("gpuUsd"), (int, float)):
        raise ValueError(f"{c['id']}: a bet needs gpuUsd")
    return c


def taken_ids(ctx: dict) -> set[str]:
    return {h["id"] for h in ctx["history"]}


def check_cards(cards, ctx: dict, slots: list[tuple[str, str]] | None) -> list[dict]:
    if not isinstance(cards, list) or (slots is not None and len(cards) != len(slots)):
        raise ValueError(f"expected {len(slots) if slots else 'a list of'} cards")
    taken, out = taken_ids(ctx), []
    for i, c in enumerate(cards):
        out.append(check_card(c, ctx, taken, slots[i] if slots else None))
        taken.add(c["id"])
    return out


def plan(cfg: dict, ctx: dict, slots: list[tuple[str, str]], ask=None) -> list[dict]:
    if not slots:
        return []
    prompt = PLAN.format(about=ABOUT, schema=CARD_SCHEMA, pack=dump(ctx), slots=dump([{"component": c, "tier": t} for c, t in slots]))
    return call(cfg, "planner", prompt, lambda a: check_cards(a["cards"], ctx, slots), ask)


def judge(cfg: dict, card: dict, ctx: dict, ask=None) -> dict:
    def check(a):
        if not isinstance(a.get("accept"), bool) or not isinstance(a.get("why"), str):
            raise ValueError("accept (bool) and why (string) are required")
        return {"accept": a["accept"], "restates": list(a.get("restates") or []), "why": a["why"]}
    mine = next(((h.get("created") or "", h["id"]) for h in ctx["history"] if h["id"] == card["id"]), ("~", card["id"]))
    # of two cards still waiting for this judge (planned in parallel), only the younger sees the older: one of them runs
    same = [h for h in ctx["history"] if h["component"] == card["component"] and h["id"] != card["id"]
            and not (h["state"] == "proposed" and ((h.get("created") or "", h["id"]) > mine))]
    return call(cfg, "judge", JUDGE.format(about=ABOUT, card=dump(card), history=dump(same)), check, ask)


def review_diff(cfg: dict, card: dict, diff: str, ask=None) -> dict:
    def check(a):
        if not isinstance(a.get("ok"), bool):
            raise ValueError("ok (bool) is required")
        return {"ok": a["ok"], "why": str(a.get("why", "")), "concerns": list(a.get("concerns") or [])}
    d = diff if len(diff) <= MAX_DIFF else diff[:MAX_DIFF] + f"\n... ({len(diff) - MAX_DIFF} more characters)"
    return call(cfg, "diffReviewer", DIFF_REVIEW.format(about=ABOUT, card=dump(card), diff=d), check, ask)


def review(cfg: dict, exp: dict, outcome: dict, ctx: dict, ask=None) -> dict:
    def check(a):
        if not str(a.get("journal", "")).strip():
            raise ValueError("journal is required")
        f = a.get("followUp")
        if f is not None:
            f = check_card(f, ctx, taken_ids(ctx))
        return {"journal": a["journal"], "followUp": f, "deadEnd": a.get("deadEnd")}
    body = {"card": exp["card"], "tier": exp["tier"], "executor": (exp.get("results") or {}).get("executor"), "outcome": outcome}
    return call(cfg, "reviewer", REVIEW.format(about=ABOUT, exp=dump(body), pack=dump(ctx), schema=CARD_SCHEMA), check, ask)


def strategy(cfg: dict, ctx: dict, frozen: list[str], slots: list[tuple[str, str]], ask=None) -> dict:
    def check(a):
        if not str(a.get("memo", "")).strip():
            raise ValueError("memo is required")
        return {"memo": a["memo"], "cards": check_cards(a.get("cards") or [], ctx, None)[: len(slots)]}
    why = "; ".join(f"{c}: {ctx['frozen'].get(c, {}).get('why', 'plateau')}" for c in frozen)
    prompt = STRATEGY.format(about=ABOUT, frozen=why, schema=CARD_SCHEMA, pack=dump(ctx), slots=dump([{"component": c, "tier": t} for c, t in slots]))
    return call(cfg, "strategist", prompt, check, ask)
