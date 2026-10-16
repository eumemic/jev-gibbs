#!/usr/bin/env python3
"""Knights-and-knaves puzzles as a test of Jev + Gibbs.

Each person is a knight (always tells the truth) or a knave (always lies). Each puzzle is generated
fresh with a unique solution, so there is no contamination and the gold labels are exact. Variables are
"is X a knight?", one Noul each.

Modes:
  screen   Per person: P(knight) from one pass over the puzzle with no types given, with everyone else's
           correct types given (oracle), and with everyone else's types random. The oracle minus one-pass
           gap is the most a Gibbs chain could gain.
  gibbs    Exact sequential Gibbs per puzzle: re-judge one person at a time given the others' current
           types, sample, repeat. Reports per-person accuracy and puzzles fully solved.

Usage:
  python kk.py screen --people 4 6 8 --puzzles 50
  python kk.py gibbs --people 6 --puzzles 50
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jev_gibbs import Jev, mean_std, metrics

HERE = Path(__file__).resolve().parent
NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
RULES = ("Every person is either a knight or a knave. Knights always tell the truth. Knaves always lie: "
         "everything a knave says is false.")
CRITERIA = {"true": "The person is a knight: what they say is true.",
            "false": "The person is a knave: what they say is false."}


# ----------------------------------------------------------------------------- puzzles

def holds(stmt: tuple, kn: dict[str, bool]) -> bool:
    kind, a, b = stmt
    if kind == "is_knight":
        return kn[a]
    if kind == "is_knave":
        return not kn[a]
    if kind == "same":
        return kn[a] == kn[b]
    return kn[a] != kn[b]  # "different"


def negate(stmt: tuple) -> tuple:
    kind, a, b = stmt
    return ({"is_knight": "is_knave", "is_knave": "is_knight", "same": "different", "different": "same"}[kind], a, b)


def say(stmt: tuple) -> str:
    kind, a, b = stmt
    return {"is_knight": f"{a} is a knight.", "is_knave": f"{a} is a knave.",
            "same": f"{a} and {b} are the same kind.", "different": f"{a} and {b} are different kinds."}[kind]


def solutions(people: list[str], statements: list[tuple[str, tuple]]) -> list[dict[str, bool]]:
    out = []
    for bits in itertools.product([True, False], repeat=len(people)):
        kn = dict(zip(people, bits))
        if all(holds(s, kn) == kn[speaker] for speaker, s in statements):
            out.append(kn)
    return out


def make_puzzle(n: int, rng: random.Random) -> dict:
    """One statement per person about others, consistent with a random assignment; resample until unique."""
    people = NAMES[:n]
    while True:
        kn = {p: rng.random() < 0.5 for p in people}
        statements = []
        for p in people:
            others = [q for q in people if q != p]
            if rng.random() < 0.5 or len(others) < 2:
                s = (rng.choice(["is_knight", "is_knave"]), rng.choice(others), None)
            else:
                a, b = rng.sample(others, 2)
                s = (rng.choice(["same", "different"]), a, b)
            if holds(s, kn) != kn[p]:  # make the statement true for knights and false for knaves
                s = negate(s)
            statements.append((p, s))
        if len(solutions(people, statements)) == 1:
            return {"people": people, "gold": kn,
                    "statements": [{"speaker": p, "says": say(s)} for p, s in statements]}


def state(puz: dict, given: dict[str, bool] | None) -> dict:
    s = {"rules": RULES, "statements": [f"{x['speaker']} says: \"{x['says']}\"" for x in puz["statements"]]}
    if given:
        s["known_types"] = {p: "knight" if v else "knave" for p, v in given.items()}
    return s


def question(person: str, given: bool) -> dict:
    ask = ("Taking `known_types` as correct, is `person` a knight?" if given else "Is `person` a knight?")
    return {"instructions": {"person": person, "ask": ask}, "criteria": CRITERIA}


# ----------------------------------------------------------------------------- modes

def screen(jev: Jev, puzzles: list[dict], workers: int) -> dict:
    def one(puz):
        rng = random.Random(zlib.crc32(json.dumps(puz["statements"]).encode()))
        row = {"one_pass": jev.nouls(state(puz, None), {p: question(p, False) for p in puz["people"]})}
        row["oracle"], row["random"] = {}, {}
        for p in puz["people"]:
            others = [q for q in puz["people"] if q != p]
            row["oracle"][p] = jev.nouls(state(puz, {q: puz["gold"][q] for q in others}), {"t": question(p, True)})["t"]
            row["random"][p] = jev.nouls(state(puz, {q: rng.random() < 0.5 for q in others}),
                                         {"t": question(p, True)})["t"]
        return row

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, puzzles))
    ys = [puz["gold"][p] for puz in puzzles for p in puz["people"]]
    out = {}
    for cond in ("one_pass", "oracle", "random"):
        ps = [r[cond][p] for r, puz in zip(rows, puzzles) for p in puz["people"]]
        solved = sum(all((r[cond][p] > 0.5) == puz["gold"][p] for p in puz["people"]) for r, puz in zip(rows, puzzles))
        out[cond] = {**metrics(ps, [int(y) for y in ys]), "puzzles_solved": solved / len(puzzles)}
    return {"summary": out, "rows": rows}


def gibbs_chain(jev: Jev, puz: dict, burn: int, keep: int, seed: int) -> dict:
    rng = random.Random(seed)
    people = puz["people"]
    cur = {p: rng.random() < 0.5 for p in people}
    p_sum = dict.fromkeys(people, 0.0)
    votes = dict.fromkeys(people, 0)
    trace = []
    for sweep in range(burn + keep):
        for p in rng.sample(people, len(people)):
            pr = jev.nouls(state(puz, {q: cur[q] for q in people if q != p}), {"t": question(p, True)})["t"]
            cur[p] = rng.random() < pr
            if sweep >= burn:
                p_sum[p] += pr
        if sweep >= burn:
            for p in people:
                votes[p] += cur[p]
        trace.append(sum(cur[p] == puz["gold"][p] for p in people))
    return {"rb": {p: p_sum[p] / keep for p in people}, "vote": {p: votes[p] / keep for p in people},
            "trace": trace}


def gibbs(jev: Jev, puzzles: list[dict], args) -> dict:
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda ip: gibbs_chain(jev, ip[1], args.burn_in, args.keep, 1000 + ip[0]),
                          enumerate(puzzles)))
    ys = [int(puz["gold"][p]) for puz in puzzles for p in puz["people"]]
    out = {}
    for est in ("rb", "vote"):
        ps = [r[est][p] for r, puz in zip(res, puzzles) for p in puz["people"]]
        solved = sum(all((r[est][p] > 0.5) == puz["gold"][p] for p in puz["people"]) for r, puz in zip(res, puzzles))
        out[est] = {**metrics(ps, ys), "puzzles_solved": solved / len(puzzles)}
    # Fraction of sweeps in which the chain's current assignment was the exact solution
    n = len(puzzles[0]["people"])
    out["at_solution"] = sum(t == n for r in res for t in r["trace"][args.burn_in:]) / (len(res) * args.keep)
    return {"summary": out, "chains": res}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["screen", "gibbs"])
    ap.add_argument("--people", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--puzzles", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--burn-in", type=int, default=10)
    ap.add_argument("--keep", type=int, default=25)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    jev = Jev("jev-latest", args.concurrency)
    results = {}
    for n in args.people:
        rng = random.Random(args.seed * 100 + n)
        puzzles = [make_puzzle(n, rng) for _ in range(args.puzzles)]
        r = screen(jev, puzzles, args.workers) if args.mode == "screen" else gibbs(jev, puzzles, args)
        results[n] = {"puzzles": puzzles, **r}
        for cond, m in r["summary"].items():
            if isinstance(m, dict):
                print(f"N={n} {cond:9s} acc {m['acc']:.3f}  puzzles solved {m['puzzles_solved']:.2f}  "
                      f"brier {m['brier']:.3f}  auroc {m['auroc']:.3f}")
            else:
                print(f"N={n} {cond:9s} {m:.3f}")
    print(f"usage: {json.dumps(jev.usage())}")
    out = HERE / "results" / f"kk_{args.mode}_{'_'.join(map(str, args.people))}.json"
    out.write_text(json.dumps({"args": vars(args), "results": results, "usage": jev.usage()}, indent=1, default=str))
    print(f"saved {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
