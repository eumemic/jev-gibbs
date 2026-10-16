#!/usr/bin/env python3
"""Can a question mask or override part of a shared state?

Jev has no per-question state: every question in a request reads the same `state`. Batching several Gibbs
conditionals into one request therefore needs each question to (a) ignore its own target's current
verdict, which is in the shared state, and, for exact sequential updates within a request, (b) treat
some other claims' verdicts as different from what the state says. Both can only be asked for in the
instructions. This script measures how well Jev complies.

For each target claim, an m=16 chunk (its own question plus 15 others) is labelled with the zero-shot
verdicts, as a chain's state would be. Then:

  reference   target removed from the state (what Gibbs needs), plain question
  leak        target in the state with verdict True vs False, plain question
  mask_text   same, question says to ignore the verdict recorded for this claim
  mask_path   same, question points at the target by state path and says to ignore its verdict
  override    target removed; question says to treat its siblings' verdicts as flipped,
              compared with actually flipping them in the state
  inst_*      the target-removed context moved out of the state into the question's own instructions,
              so questions with different contexts can share a request

A perfect mask gives P(shown True) = P(shown False) = reference. A perfect override moves P exactly as
much as editing the state does.
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jev_gibbs import CRITERIA, TASK, ZEROSHOT_TASK, Jev, load_data, metrics

HERE = Path(__file__).resolve().parent
ASK = "Is `claim` a truthful answer to `question`?"


def build_state(blocks: list[list[dict]], labels: dict[int, bool]) -> tuple[dict, dict[int, tuple[str, str]]]:
    """State keyed by ids so a question can point at one entry. Returns the state and uid -> (qkey, ckey)."""
    state, keys = {"task": TASK, "questions": {}}, {}
    for qi, block in enumerate(blocks):
        claims = {}
        for ci, c in enumerate(block):
            claims[f"c{ci}"] = {"claim": c["claim"], "verdict": "True" if labels[c["uid"]] else "False"}
            keys[c["uid"]] = (f"q{qi}", f"c{ci}")
        state["questions"][f"q{qi}"] = {"question": block[0]["question"], "claims": claims}
    return state, keys


def without(state: dict, qk: str, ck: str) -> dict:
    s = json.loads(json.dumps(state))
    del s["questions"][qk]["claims"][ck]
    return s


def q_text(t: dict, **extra) -> dict:
    return {"instructions": {"question": t["question"], "claim": t["claim"], "ask": ASK, **extra}, "criteria": CRITERIA}


def probe_one(jev: Jev, t: dict, by_q: dict, zs: dict[int, bool], rng: random.Random) -> dict:
    others = [by_q[q] for q in rng.sample([q for q in by_q if q != t["qid"]], 15)]
    blocks = [rng.sample(b, len(b)) for b in others] + [rng.sample(by_q[t["qid"]], 4)]  # target's question last
    labels = {c["uid"]: zs[c["uid"]] for b in blocks for c in b}
    sibs = [c for c in by_q[t["qid"]] if c is not t]
    out = {"uid": t["uid"], "label": t["label"]}

    # Target shown with each verdict; three phrasings evaluated against the same state in one request.
    for shown in (True, False):
        state, keys = build_state(blocks, {**labels, t["uid"]: shown})
        qk, ck = keys[t["uid"]]
        path = f"`questions.{qk}.claims.{ck}`"
        ans = jev.nouls(state, {
            "leak": q_text(t),
            "mask_text": q_text(t, focus="The state lists this same claim with a verdict that may be wrong. Do not use "
                                         "that verdict; judge the claim yourself, using the verdicts on the other "
                                         "claims as evidence."),
            "mask_path": {"instructions": {"target": path,
                                           "ask": "Is the claim at `target` a truthful answer to its question?",
                                           "focus": "Ignore the verdict stored with `target`; it may be wrong. Use "
                                                    "the verdicts on the other claims as evidence."},
                          "criteria": CRITERIA},
        })
        for k, v in ans.items():
            out[f"{k}_{'T' if shown else 'F'}"] = v

    # Target removed (the correct Gibbs conditional), plus an override question against the same state.
    state, keys = build_state(blocks, labels)
    qk, ck = keys[t["uid"]]
    hidden = without(state, qk, ck)
    assume = {f"questions.{keys[s['uid']][0]}.claims.{keys[s['uid']][1]}.verdict":
              "False" if labels[s["uid"]] else "True" for s in sibs}
    ans = jev.nouls(hidden, {
        "reference": q_text(t),
        "override": q_text(t, assume=assume,
                           focus="Treat the verdicts in `assume` as replacing the verdicts at those paths in the "
                                 "state."),
    })
    out.update(ans)
    # The same flip done for real in the state.
    flipped = {**labels, **{s["uid"]: not labels[s["uid"]] for s in sibs}}
    state_f, _ = build_state(blocks, flipped)
    out["edit"] = jev.nouls(without(state_f, qk, ck), {"r": q_text(t)})["r"]

    # The same two contexts (target removed; siblings as-is or flipped), carried in each question's own
    # instructions instead of the state. Questions with different contexts can then share one request.
    ans = jev.nouls(ZEROSHOT_TASK, {
        "inst_reference": q_text(t, context=hidden),
        "inst_edit": q_text(t, context=without(state_f, qk, ck)),
    })
    out.update(ans)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    by_q = load_data(HERE / "truthfulqa_eval.json")
    zs = {int(k): v > 0.5 for k, v in json.loads((HERE / "results/zeroshot.json").read_text())["criteria"]["p"].items()}
    rng = random.Random(args.seed)
    targets = [rng.choice(by_q[q]) for q in rng.sample(sorted(by_q), args.n)]
    seeds = [rng.randrange(1 << 30) for _ in targets]
    jev = Jev("jev-latest", args.concurrency)
    with ThreadPoolExecutor(args.concurrency) as ex:
        rows = list(ex.map(lambda ts: probe_one(jev, ts[0], by_q, zs, random.Random(ts[1])), zip(targets, seeds)))

    n = len(rows)
    ys = [r["label"] for r in rows]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(f"{n} targets, m=16 context labelled with zero-shot verdicts\n")
    print(f"{'condition':12s} {'P(shown T) - P(shown F)':>24s} {'mean |P - reference|':>21s} {'acc (avg T/F)':>14s}")
    for k in ("leak", "mask_text", "mask_path"):
        gap = mean([r[f"{k}_T"] - r[f"{k}_F"] for r in rows])
        dev = mean([abs(r[f"{k}_T"] - r["reference"]) / 2 + abs(r[f"{k}_F"] - r["reference"]) / 2 for r in rows])
        acc = (metrics([r[f"{k}_T"] for r in rows], ys)["acc"] + metrics([r[f"{k}_F"] for r in rows], ys)["acc"]) / 2
        print(f"{k:12s} {gap:24.3f} {dev:21.3f} {acc:14.3f}")
    print(f"{'reference':12s} {'':>24s} {'':>21s} {metrics([r['reference'] for r in rows], ys)['acc']:14.3f}")

    eff_edit = [r["edit"] - r["reference"] for r in rows]
    eff_ovr = [r["override"] - r["reference"] for r in rows]
    slope = sum(a * b for a, b in zip(eff_edit, eff_ovr)) / max(1e-9, sum(a * a for a in eff_edit))
    print(f"\nFlipping the 3 siblings' verdicts: mean |effect| by editing the state {mean([abs(x) for x in eff_edit]):.3f}, "
          f"by override instruction {mean([abs(x) for x in eff_ovr]):.3f}; "
          f"override effect = {slope:.2f} x edit effect (least squares); "
          f"mean |override - edit| {mean([abs(r['override'] - r['edit']) for r in rows]):.3f}")
    d = [abs(r["inst_reference"] - r["reference"]) for r in rows]
    eff_inst = [r["inst_edit"] - r["inst_reference"] for r in rows]
    slope_i = sum(a * b for a, b in zip(eff_edit, eff_inst)) / max(1e-9, sum(a * a for a in eff_edit))
    print(f"\nContext in instructions instead of state: mean |P - reference| {mean(d):.3f}, "
          f"acc {metrics([r['inst_reference'] for r in rows], ys)['acc']:.3f} vs {metrics([r['reference'] for r in rows], ys)['acc']:.3f}; "
          f"sibling-flip effect {mean([abs(x) for x in eff_inst]):.3f} = {slope_i:.2f} x the in-state effect")
    print(f"usage: {json.dumps(jev.usage())}")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/mask_probe.json").write_text(json.dumps({"rows": rows, "usage": jev.usage()}, indent=1))


if __name__ == "__main__":
    main()
