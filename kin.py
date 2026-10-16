#!/usr/bin/env python3
"""Kinship chains (in the style of CLUTRR) as a test of Jev + Gibbs.

Each story comes from a fresh random family tree (three generations, in-laws, one marriage between the two
families). A random path of k relatives P0..Pk is drawn through the tree, and the story states only the k
consecutive relations, in varied phrasings and shuffled order. The variables are the relations between every
pair (Pi, Pj) with j - i >= 2 whose relation has a single kinship term, one Choice question each over 23
terms. Getting (P0, Pk) in one pass means composing k relations. Given the neighbouring relations, each one
is a single composition step.

Modes:
  screen   Per variable: one pass over the story; with every other variable's correct relation given
           (oracle); with random relations given; with Jev's own one-pass answers given (one refinement step).
  gibbs    Exact sequential Gibbs per story, started from a sample of the one-pass answers.

Usage:
  python3 kin.py screen --k 2 3 4 6 --stories 40
  python3 kin.py gibbs --k 4 6 --stories 40
"""
from __future__ import annotations

import argparse
import json
import math
import random
import threading
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jev_gibbs import Jev, mean_std

HERE = Path(__file__).resolve().parent
FEMALE = ["Alice", "Beth", "Clara", "Diana", "Emma", "Fiona", "Grace", "Hannah", "Iris", "Julia", "Kate", "Laura",
          "Maya", "Nora", "Olivia", "Paula", "Penny", "Rosa", "Sara", "Tina", "Uma", "Vera", "Wendy", "Zoe"]
MALE = ["Adam", "Ben", "Carl", "David", "Ethan", "Frank", "George", "Henry", "Ivan", "Jack", "Kevin", "Leo",
        "Mark", "Noah", "Oscar", "Peter", "Ray", "Sam", "Tom", "Victor", "Will", "Xavier", "Yusuf", "Zach"]
TERMS = {  # kinship term -> definition shown to Jev
    "father": "male parent", "mother": "female parent", "son": "male child", "daughter": "female child",
    "brother": "male sibling", "sister": "female sibling", "husband": "male spouse", "wife": "female spouse",
    "grandfather": "father of a parent", "grandmother": "mother of a parent",
    "grandson": "son of a child", "granddaughter": "daughter of a child",
    "uncle": "brother of a parent, or husband of a parent's sister",
    "aunt": "sister of a parent, or wife of a parent's brother",
    "nephew": "son of a sibling, or son of a spouse's sibling",
    "niece": "daughter of a sibling, or daughter of a spouse's sibling",
    "cousin": "child of a parent's sibling",
    "father-in-law": "father of a spouse", "mother-in-law": "mother of a spouse",
    "son-in-law": "husband of a child", "daughter-in-law": "wife of a child",
    "brother-in-law": "brother of a spouse, or husband of a sibling",
    "sister-in-law": "sister of a spouse, or wife of a sibling",
}
PLACES = ["park", "museum", "beach", "zoo", "market", "cinema"]
VERBS = ["called", "visited", "had lunch with", "went hiking with", "wrote a letter to"]


# ----------------------------------------------------------------------------- family trees

class Tree:
    def __init__(self, rng: random.Random):
        self.g: dict[str, str] = {}  # name -> "F" or "M"
        self.parents: dict[str, list[str]] = {}
        self.children: dict[str, list[str]] = {}
        self.spouse: dict[str, str] = {}
        names = {"F": rng.sample(FEMALE, len(FEMALE)), "M": rng.sample(MALE, len(MALE))}

        def person(sex, parents=()):
            n = names[sex].pop()
            self.g[n], self.parents[n], self.children[n] = sex, list(parents), []
            for p in parents:
                self.children[p].append(n)
            return n

        def marry(a, b):
            self.spouse[a], self.spouse[b] = b, a

        couples = []
        for _ in range(2):  # generation 0: two unrelated couples
            a, b = person("M"), person("F")
            marry(a, b)
            couples.append((a, b))
        gen1 = [[person(rng.choice("FM"), c) for _ in range(3)] for c in couples]
        # one marriage across the two families, if the sexes allow it
        x = next((m for m in gen1[0] if any(self.g[w] != self.g[m] for w in gen1[1])), None)
        if x:
            y = next(w for w in gen1[1] if self.g[w] != self.g[x])
            marry(x, y)
        new_couples = [tuple(sorted((x, y)))] if x else []
        for m in gen1[0] + gen1[1]:
            if m not in self.spouse:  # everyone else in generation 1 marries an outsider
                o = person("F" if self.g[m] == "M" else "M")
                marry(m, o)
                new_couples.append((m, o))
        for c in new_couples:  # generation 2
            for _ in range(rng.choice([1, 2, 2])):
                person(rng.choice("FM"), c)

    def sibs(self, x):
        return {c for p in self.parents[x] for c in self.children[p]} - {x}

    def relation(self, x: str, y: str) -> str | None:
        """The single kinship term for 'x is y's ___', or None if there is none or more than one."""
        f = self.g[x] == "F"
        pick = lambda fem, masc: fem if f else masc  # noqa: E731
        par, ch, sp = self.parents[y], self.children[y], self.spouse.get(y)
        found = set()
        if x in par:
            found.add(pick("mother", "father"))
        if x in ch:
            found.add(pick("daughter", "son"))
        if x in self.sibs(y):
            found.add(pick("sister", "brother"))
        if x == sp:
            found.add(pick("wife", "husband"))
        if any(x in self.parents[p] for p in par):
            found.add(pick("grandmother", "grandfather"))
        if any(x in self.children[c] for c in ch):
            found.add(pick("granddaughter", "grandson"))
        if any(x in self.sibs(p) or x in {self.spouse.get(s) for s in self.sibs(p)} for p in par):
            found.add(pick("aunt", "uncle"))
        if any(x in self.children[s] for s in self.sibs(y)) or (sp and any(x in self.children[s] for s in self.sibs(sp))):
            found.add(pick("niece", "nephew"))
        if any(x in self.children[s] for p in par for s in self.sibs(p)):
            found.add("cousin")
        if sp and x in self.parents[sp]:
            found.add(pick("mother-in-law", "father-in-law"))
        if any(x == self.spouse.get(c) for c in ch):
            found.add(pick("daughter-in-law", "son-in-law"))
        if (sp and x in self.sibs(sp)) or x in {self.spouse.get(s) for s in self.sibs(y)}:
            found.add(pick("sister-in-law", "brother-in-law"))
        return found.pop() if len(found) == 1 else None

    def neighbours(self, x):
        return set(self.parents[x]) | set(self.children[x]) | self.sibs(x) | ({self.spouse[x]} if x in self.spouse else set())


# ----------------------------------------------------------------------------- stories

def sentence(tree: Tree, x: str, y: str, rng: random.Random) -> str:
    """State 'x is y's <basic relation>' in one of several phrasings."""
    w = tree.relation(x, y)  # mother/father/son/daughter/brother/sister/wife/husband
    inv = tree.relation(y, x)
    xp, yp = ("her" if tree.g[x] == "F" else "his"), ("her" if tree.g[y] == "F" else "his")
    return rng.choice([
        f"{x} is {y}'s {w}.",
        f"{y}'s {w} is {x}.",
        f"{x} took {xp} {inv} {y} to the {rng.choice(PLACES)}.",
        f"{y} {rng.choice(VERBS)} {yp} {w} {x}.",
    ])


def make_story(k: int, rng: random.Random) -> dict:
    while True:
        tree = Tree(rng)
        people = list(tree.g)
        path = [rng.choice(people)]
        while len(path) < k + 1:
            nxt = sorted(n for n in tree.neighbours(path[-1]) if n not in path)  # sorted: set order varies by process
            if not nxt:
                break
            path.append(rng.choice(nxt))
        if len(path) < k + 1 or tree.relation(path[0], path[-1]) is None:
            continue
        if any(tree.relation(a, b) is None for a, b in zip(path, path[1:])):
            continue
        variables = {f"{i}|{j}": tree.relation(path[i], path[j]) for i in range(k + 1) for j in range(i + 2, k + 1)}
        variables = {v: r for v, r in variables.items() if r is not None}
        sents = [sentence(tree, a, b, rng) if rng.random() < 0.5 else sentence(tree, b, a, rng)
                 for a, b in zip(path, path[1:])]
        rng.shuffle(sents)
        return {"k": k, "path": path, "story": " ".join(sents), "gold": variables}


def names(story: dict, v: str) -> tuple[str, str]:
    i, j = map(int, v.split("|"))
    return story["path"][i], story["path"][j]


# How labels are shown to a conditional (set from the command line):
#   "correct"    known_relations, to be taken as correct (right for the oracle, wrong for a chain)
#   "tentative"  tentative_relations, earlier guesses that may be wrong; the story stays authoritative
LABELS = "correct"
RELEVANT_ONLY = False  # show only labels that involve one of the two people asked about
LABEL_KEY = {"correct": "known_relations", "tentative": "tentative_relations"}


def state(story: dict, given: dict[str, str] | None = None, skip: str | None = None) -> dict:
    s = {"story": story["story"]}
    if given is not None:
        ends = set(skip.split("|")) if (RELEVANT_ONLY and skip) else None
        s[LABEL_KEY[LABELS]] = [f"{names(story, v)[0]} is {names(story, v)[1]}'s {r}."
                                for v, r in given.items() if v != skip and (ends is None or ends & set(v.split("|")))]
    return s


def question(story: dict, v: str, given: bool, rng: random.Random) -> dict:
    x, y = names(story, v)
    ask = f"In this family, {x} is {y}'s what?"
    if given and LABELS == "correct":
        ask = f"Taking `known_relations` as correct: {ask}"
    elif given:
        ask = (f"{ask} The story is authoritative. `tentative_relations` are earlier guesses that may be wrong; "
               "use them only where the story supports them.")
    opts = list(TERMS)
    # Jev can lean toward the first option, so the order is shuffled: by rng, or (rng=None) fixed per
    # story and variable so that every method sees the same order.
    (rng or random.Random(zlib.crc32(f"{story['story']}|{v}".encode()))).shuffle(opts)
    return {"instructions": ask, "criteria": {o: TERMS[o] for o in opts}}


def argmax(p: dict[str, float]) -> str:
    return max(p, key=p.get)


# ----------------------------------------------------------------------------- modes

def screen(jev: Jev, stories: list[dict], workers: int) -> dict:
    def one(si_story):
        si, st = si_story
        rng = random.Random(si)
        one_pass = jev.choices(state(st), {v: question(st, v, False, rng) for v in st["gold"]})
        own = {v: argmax(p) for v, p in one_pass.items()}
        row = {"one_pass": one_pass, "oracle": {}, "random": {}, "self": {}}
        for v in st["gold"]:
            rand = {u: rng.choice(list(TERMS)) for u in st["gold"]}
            for cond, given in (("oracle", st["gold"]), ("random", rand), ("self", own)):
                row[cond][v] = jev.choices(state(st, given, skip=v), {"t": question(st, v, True, rng)})["t"]
        return row

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, enumerate(stories)))
    return {"rows": rows, "summary": summarise(stories, rows, ("one_pass", "oracle", "random", "self"))}


def summarise(stories, rows, conds) -> dict:
    out = {}
    for cond in conds:
        allv = [(st, v) for st in stories for v in st["gold"]]
        acc = sum(argmax(r[cond][v]) == st["gold"][v] for st, r in zip(stories, rows) for v in st["gold"]) / len(allv)
        end = sum(argmax(r[cond][f"0|{st['k']}"]) == st["gold"][f"0|{st['k']}"] for st, r in zip(stories, rows)) / len(stories)
        solved = sum(all(argmax(r[cond][v]) == st["gold"][v] for v in st["gold"]) for st, r in zip(stories, rows)) / len(stories)
        p_gold = sum(r[cond][v].get(st["gold"][v], 0) for st, r in zip(stories, rows) for v in st["gold"]) / len(allv)
        out[cond] = {"acc": round(acc, 3), "end_to_end": round(end, 3), "stories_solved": round(solved, 3),
                     "mean_p_gold": round(p_gold, 3), "n_vars": len(allv)}
    return out


def draw(p: dict[str, float], T: float, rng: random.Random) -> str:
    """Sample an option from p sharpened to temperature T (T=1: Jev's probabilities; T->0: argmax)."""
    w = [max(x, 1e-6) ** (1 / T) for x in p.values()]
    return rng.choices(list(p), weights=w)[0]


def gibbs_chain(jev: Jev, st: dict, one_pass: dict, burn: int, keep: int, seed: int, T: float = 1.0,
                fixed_order: bool = False) -> dict:
    rng = random.Random(seed)
    vs = list(st["gold"])
    cur = {v: draw(one_pass[v], T, rng) for v in vs}
    p_sum = {v: Counter() for v in vs}
    votes = {v: Counter() for v in vs}
    at_truth = []
    for sweep in range(burn + keep):
        for v in rng.sample(vs, len(vs)):
            p = jev.choices(state(st, cur, skip=v), {"t": question(st, v, True, None if fixed_order else rng)})["t"]
            cur[v] = draw(p, T, rng)
            if sweep >= burn:
                p_sum[v].update(p)
        if sweep >= burn:
            for v in vs:
                votes[v][cur[v]] += 1
            at_truth.append(all(cur[v] == st["gold"][v] for v in vs))
    return {"rb": {v: {o: x / keep for o, x in p_sum[v].items()} for v in vs},
            "vote": {v: {o: x / keep for o, x in votes[v].items()} for v in vs},
            "at_truth": sum(at_truth) / keep}


def gibbs(jev: Jev, stories: list[dict], args) -> dict:
    def run(si_story):
        si, st = si_story
        rng = random.Random(si)
        one_pass = jev.choices(state(st), {v: question(st, v, False, rng) for v in st["gold"]})
        return {"one_pass": one_pass, **gibbs_chain(jev, st, one_pass, args.burn_in, args.keep, 7000 + si,
                                                    args.temperature)}

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run, enumerate(stories)))
    s = summarise(stories, rows, ("one_pass", "rb", "vote"))
    s["at_truth"] = round(sum(r["at_truth"] for r in rows) / len(rows), 3)
    return {"rows": rows, "summary": s}


def sign_test(b: int, c: int) -> float:
    """Exact two-sided sign test (McNemar) on b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def compare(jev: Jev, stories: list[dict], args, k: int) -> dict:
    """One pass, one refinement step and Gibbs at several temperatures on the same stories. Every method
    starts from the same one-pass answers and sees the same option order. Results are cached per story and
    method in a JSONL file, so an interrupted run resumes where it stopped."""
    cache = HERE / "results" / f"kin_compare_k{k}_s{args.seed}_b{args.burn_in}_n{args.keep}.jsonl"
    done: dict[tuple[str, int], dict] = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            rec = json.loads(line)
            done[(rec["method"], rec["si"])] = rec
    lock, errors = threading.Lock(), []

    def save(rec):
        with lock:
            done[(rec["method"], rec["si"])] = rec
            with cache.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    def one_pass(si):
        st = stories[si]
        save({"method": "one_pass", "si": si,
              "p": jev.choices(state(st), {v: question(st, v, False, None) for v in st["gold"]})})

    def refine(si):  # every variable re-judged once, given Jev's best one-pass guesses for the others
        st, op = stories[si], done[("one_pass", si)]["p"]
        own = {v: argmax(p) for v, p in op.items()}
        save({"method": "refine1", "si": si,
              "p": {v: jev.choices(state(st, own, skip=v), {"t": question(st, v, True, None)})["t"] for v in st["gold"]}})

    def chain(si, T):
        st, op = stories[si], done[("one_pass", si)]["p"]
        save({"method": f"gibbs_T{T:g}", "si": si,
              **gibbs_chain(jev, st, op, args.burn_in, args.keep, 7000 + si, T, fixed_order=True)})

    def safe(job):
        try:
            job[0](*job[1])
        except Exception as e:  # keep going; failed jobs are retried on the next run
            with lock:
                errors.append(f"{job[0].__name__}{job[1]}: {e!r}"[:300])

    n = len(stories)
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(safe, [(one_pass, (si,)) for si in range(n) if ("one_pass", si) not in done]))
    jobs = [(refine, (si,)) for si in range(n) if ("refine1", si) not in done and ("one_pass", si) in done]
    jobs += [(chain, (si, T)) for T in args.temperatures for si in range(n)
             if (f"gibbs_T{T:g}", si) not in done and ("one_pass", si) in done]
    random.Random(0).shuffle(jobs)  # mix short and long jobs across workers
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(safe, jobs))

    methods = ["one_pass", "refine1"] + [f"gibbs_T{T:g}" for T in args.temperatures]
    complete = [si for si in range(n) if all((m, si) in done for m in methods)]
    sts = [stories[si] for si in complete]

    def probs(m, si, est="rb"):
        rec = done[(m, si)]
        return rec["p"] if "p" in rec else rec[est]

    cols = [("one_pass", "rb"), ("refine1", "rb")] + [(f"gibbs_T{T:g}", e) for T in args.temperatures for e in ("rb", "vote")]
    calls_per_story = {"one_pass": 1, "refine1": sum(len(st["gold"]) for st in sts) / max(1, len(sts))}
    for T in args.temperatures:
        calls_per_story[f"gibbs_T{T:g}"] = calls_per_story["refine1"] * (args.burn_in + args.keep)

    def correct(m, est):
        per_var = {(si, v): argmax(probs(m, si, est)[v]) == stories[si]["gold"][v] for si in complete for v in stories[si]["gold"]}
        end = {si: per_var[(si, f"0|{k}")] for si in complete}
        return per_var, end

    base = {name: correct(name, "rb") for name in ("one_pass", "refine1")}
    summary = {}
    for m, est in cols:
        per_var, end = correct(m, est)
        row = {"acc": sum(per_var.values()) / len(per_var), "end_to_end": sum(end.values()) / len(end),
               "stories_solved": sum(all(per_var[(si, v)] for v in stories[si]["gold"]) for si in complete) / len(complete),
               "mean_p_gold": sum(probs(m, si, est)[v].get(stories[si]["gold"][v], 0) for si in complete
                                  for v in stories[si]["gold"]) / len(per_var),
               "calls_per_story": calls_per_story[m]}
        if m.startswith("gibbs"):
            row["at_truth"] = sum(done[(m, si)]["at_truth"] for si in complete) / len(complete)
        for bname, (bvar, bend) in base.items():
            if bname == m:
                continue
            b = sum(end[si] and not bend[si] for si in complete)
            c = sum(bend[si] and not end[si] for si in complete)
            bv = sum(per_var[x] and not bvar[x] for x in per_var)
            cv = sum(bvar[x] and not per_var[x] for x in per_var)
            row[f"vs_{bname}"] = {"end_to_end": [b, c, round(sign_test(b, c), 4)],
                                  "per_variable": [bv, cv, round(sign_test(bv, cv), 4)]}
        summary[f"{m}[{est}]" if m.startswith("gibbs") else m] = row
    return {"complete_stories": len(complete), "errors": errors, "summary": summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["screen", "gibbs", "compare"])
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--stories", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--burn-in", type=int, default=5)
    ap.add_argument("--keep", type=int, default=20)
    ap.add_argument("--labels", choices=["correct", "tentative"], default=None,
                    help="how other variables' labels are presented to a conditional "
                         "(default: tentative for compare, correct otherwise)")
    ap.add_argument("--relevant-only", action="store_true", help="show only labels involving the two people")
    ap.add_argument("--temperature", type=float, default=1.0, help="gibbs: sample from p^(1/T)")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.3, 0.1], help="compare: chains to run")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=40)
    args = ap.parse_args()
    global LABELS, RELEVANT_ONLY
    LABELS = args.labels or ("tentative" if args.mode == "compare" else "correct")
    RELEVANT_ONLY = args.relevant_only

    jev = Jev("jev-latest", args.concurrency)
    results = {}
    for k in args.k:
        rng = random.Random(args.seed * 1000 + k)
        stories = [make_story(k, rng) for _ in range(args.stories)]
        if args.mode == "compare":
            r = compare(jev, stories, args, k)
            results[k] = {"stories": stories, **r}
            print(f"k={k}: {r['complete_stories']} stories complete, {len(r['errors'])} failed jobs", flush=True)
            for e in r["errors"][:3]:
                print(f"   {e}")
            print(f"  {'method':18s} {'per var':>8s} {'end2end':>8s} {'solved':>7s} {'P(gold)':>8s} {'calls/story':>12s}  "
                  f"end-to-end vs one pass (+/-, p)   vs one step (+/-, p)")
            for name, m in r["summary"].items():
                vo, vr = m.get("vs_one_pass"), m.get("vs_refine1")
                fmt = lambda x: f"+{x['end_to_end'][0]}/-{x['end_to_end'][1]} p={x['end_to_end'][2]:.3f}" if x else "-"  # noqa: E731
                print(f"  {name:18s} {m['acc']:8.3f} {m['end_to_end']:8.3f} {m['stories_solved']:7.3f} {m['mean_p_gold']:8.3f} "
                      f"{m['calls_per_story']:12.0f}  {fmt(vo):30s} {fmt(vr)}", flush=True)
            continue
        r = screen(jev, stories, args.workers) if args.mode == "screen" else gibbs(jev, stories, args)
        results[k] = {"stories": stories, **r}
        for cond, m in r["summary"].items():
            if isinstance(m, dict):
                print(f"k={k} {cond:9s} acc {m['acc']:.3f}  end-to-end {m['end_to_end']:.3f}  "
                      f"stories solved {m['stories_solved']:.3f}  mean P(gold) {m['mean_p_gold']:.3f}  "
                      f"({m['n_vars']} variables)", flush=True)
            else:
                print(f"k={k} {cond:9s} {m}", flush=True)
    print(f"usage: {json.dumps(jev.usage())}")
    out = HERE / "results" / f"kin_{args.mode}_k{'_'.join(map(str, args.k))}{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps({"args": vars(args), "results": results, "usage": jev.usage()}, indent=1))
    print(f"saved {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
