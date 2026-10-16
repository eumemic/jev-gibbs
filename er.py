#!/usr/bin/env python3
"""Entity resolution on MusicBrainz 20K (Leipzig) as a test of Jev + Gibbs.

A block is one song cluster with 3–5 records plus the clusters that look most like it (shared title,
artist or album words), capped at --block-size records. Variables are all pairs of records from different
sources in a block ("same track?", one Noul each). Same-source pairs are excluded: each source is
deduplicated, so they are known non-matches.

screen mode, per pair:
  alone      only the two records in the state
  context    the whole block in the state, no decisions
  self       the whole block plus Jev's own context-pass decisions on every other pair (one refinement step)
  oracle     the whole block plus the correct decisions on every other pair
plus code baselines on the context probabilities: threshold at 0.5, transitive closure, and correlation
clustering (greedy local search on log-odds, records from one source kept apart).

Usage:
  python3 -I er.py screen --blocks 60
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with python3 -I; keep local imports working
from jev_gibbs import Jev, metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data/musicbrainz/musicbrainz-20-A01.csv.dapo"
FIELDS = ("number", "title", "length", "artist", "album", "year", "language")
TASK = ("Song records from several music databases. One track can appear in several databases with different "
        "formatting, misspellings, missing values, or values in the wrong field.")
CRITERIA = {
    "true": "Both records describe the same track (the same song on the same release), even if fields are "
            "formatted differently, misspelled, missing, or in the wrong field.",
    "false": "They describe different tracks, such as different songs by the same artist or on the same album.",
}


# ----------------------------------------------------------------------------- data and blocks

def load_records(path: Path = DATA) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tokens(r: dict) -> set[str]:
    words = re.findall(r"\w+", f"{r['title']} {r['artist']} {r['album']}".lower())
    return {t for t in words if len(t) > 1 and not t.isdigit()}  # numbers (track, year) would block on noise


def make_blocks(records: list[dict], n_blocks: int, max_size: int, seed: int) -> list[dict]:
    by_cid = collections.defaultdict(list)
    for r in records:
        by_cid[r["CID"]].append(r)
    toks = {r["TID"]: tokens(r) for r in records}
    df = collections.Counter(t for ts in toks.values() for t in ts)
    index = collections.defaultdict(set)
    for r in records:
        for t in toks[r["TID"]]:
            if df[t] <= 200:  # skip words too common to block on
                index[t].add(r["TID"])
    tid = {r["TID"]: r for r in records}
    rng = random.Random(seed)
    seeds = [c for c, rs in by_cid.items() if len(rs) >= 3]
    rng.shuffle(seeds)
    blocks, used = [], set()
    for c0 in seeds:
        if len(blocks) == n_blocks:
            break
        if c0 in used:
            continue
        score = collections.Counter()  # other cluster -> best token Jaccard with any seed record
        for r in by_cid[c0]:
            cand = {x for t in toks[r["TID"]] if t in index for x in index[t]}
            for x in cand:
                c = tid[x]["CID"]
                if c == c0:
                    continue
                a, b = toks[r["TID"]], toks[x]
                score[c] = max(score[c], len(a & b) / max(1, len(a | b)))
        members = list(by_cid[c0])
        for c, _ in score.most_common():
            if c in used or len(members) + len(by_cid[c]) > max_size:
                continue
            members += by_cid[c]
            used.add(c)
        if len(members) < len(by_cid[c0]) + 2:
            continue  # too few look-alikes to make a block
        used.add(c0)
        rng.shuffle(members)
        recs = {f"r{i + 1}": m for i, m in enumerate(members)}
        pairs = [(a, b) for i, a in enumerate(recs) for b in list(recs)[i + 1:]
                 if recs[a]["SourceID"] != recs[b]["SourceID"]]
        blocks.append({"seed_cluster": c0, "records": {k: {f: v[f] for f in FIELDS} for k, v in recs.items()},
                       "cid": {k: v["CID"] for k, v in recs.items()},
                       "source": {k: v["SourceID"] for k, v in recs.items()},
                       "pairs": pairs, "gold": {f"{a}|{b}": int(recs[a]["CID"] == recs[b]["CID"]) for a, b in pairs}})
    return blocks


# ----------------------------------------------------------------------------- prompts

def block_state(block: dict, keys=None, decisions: dict[str, int] | None = None, skip: str | None = None) -> dict:
    recs = block["records"] if keys is None else {k: block["records"][k] for k in keys}
    s = {"task": TASK, "records": recs}
    if decisions is not None:
        same = [k.split("|") for k, v in decisions.items() if k != skip and v]
        diff = [k.split("|") for k, v in decisions.items() if k != skip and not v]
        s["decisions"] = {"same_track": same, "different_tracks": diff}
    return s


def pair_question(a: str, b: str, given: bool) -> dict:
    ask = f"Do `records.{a}` and `records.{b}` describe the same track?"
    if given:
        ask = "Taking `decisions` as correct, d" + ask[1:]
    return {"instructions": ask, "criteria": CRITERIA}


# ----------------------------------------------------------------------------- code baselines

def components(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    roots = {}
    return {n: roots.setdefault(find(n), len(roots)) for n in nodes}


def correlation_clustering(block: dict, p: dict[str, float], iters: int = 50) -> dict[str, int]:
    """Greedy local search maximising the sum of log-odds over same-cluster pairs; one source per cluster."""
    nodes = list(block["records"])
    w = {}
    for k, v in p.items():
        v = min(max(v, 1e-3), 1 - 1e-3)
        a, b = k.split("|")
        w[(a, b)] = w[(b, a)] = math.log(v / (1 - v))
    lab = components(nodes, [tuple(k.split("|")) for k, v in p.items() if v > 0.5])
    for _ in range(iters):
        moved = False
        for n in nodes:
            cur = lab[n]
            groups = collections.defaultdict(list)
            for m in nodes:
                if m != n:
                    groups[lab[m]].append(m)
            # Joining a group scores the summed log-odds to its members; a group can't hold two records
            # from one source. Being alone scores 0 (keeping the current label if already alone).
            options = {g: sum(w.get((n, m), 0) for m in ms) for g, ms in groups.items()
                       if all(block["source"][m] != block["source"][n] for m in ms)}
            options[cur if cur not in groups else max(lab.values()) + 1] = 0.0
            best = max(options, key=lambda g: (options[g], g == cur))
            if best != cur and not (cur in options and options[cur] >= options[best] - 1e-9):
                lab[n], moved = best, True
        if not moved:
            break
    return lab


def pair_scores(block: dict, lab: dict[str, int]) -> dict[str, float]:
    return {f"{a}|{b}": float(lab[a] == lab[b]) for a, b in block["pairs"]}


def prf(pred: list[bool], gold: list[int]) -> dict:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(2 * pr * rc / max(1e-9, pr + rc), 3)}


# ----------------------------------------------------------------------------- screen

def screen(jev: Jev, blocks: list[dict], workers: int) -> dict:
    def ctx(block):
        return jev.nouls(block_state(block), {f"{a}|{b}": pair_question(a, b, False) for a, b in block["pairs"]})

    with ThreadPoolExecutor(workers) as ex:
        p_ctx = list(ex.map(ctx, blocks))

    jobs = []
    for bi, block in enumerate(blocks):
        own = {k: int(v > 0.5) for k, v in p_ctx[bi].items()}
        for a, b in block["pairs"]:
            k = f"{a}|{b}"
            jobs.append((bi, k, "alone", block_state(block, keys=[a, b]), pair_question(a, b, False)))
            jobs.append((bi, k, "self", block_state(block, decisions=own, skip=k), pair_question(a, b, True)))
            jobs.append((bi, k, "oracle", block_state(block, decisions=block["gold"], skip=k), pair_question(a, b, True)))
    with ThreadPoolExecutor(workers) as ex:
        outs = list(ex.map(lambda j: (j[0], j[1], j[2], jev.nouls(j[3], {"t": j[4]})["t"]), jobs))
    p = {"context": p_ctx, "alone": [{} for _ in blocks], "self": [{} for _ in blocks], "oracle": [{} for _ in blocks]}
    for bi, k, cond, v in outs:
        p[cond][bi][k] = v

    gold = [block["gold"][f"{a}|{b}"] for block in blocks for a, b in block["pairs"]]
    summary = {}

    def add(name, scores):
        flat = [scores[bi][f"{a}|{b}"] for bi, block in enumerate(blocks) for a, b in block["pairs"]]
        summary[name] = {**{k: v for k, v in metrics(flat, gold).items() if k in ("acc", "brier", "auroc")},
                         **prf([x > 0.5 for x in flat], gold)}
    for cond in ("alone", "context", "self", "oracle"):
        add(f"jev {cond}", p[cond])
    for src in ("alone", "context"):
        add(f"{src} + transitive closure", [pair_scores(b, components(list(b["records"]),
            [tuple(k.split("|")) for k, v in p[src][bi].items() if v > 0.5])) for bi, b in enumerate(blocks)])
        add(f"{src} + correlation clustering", [pair_scores(b, correlation_clustering(b, p[src][bi]))
                                                for bi, b in enumerate(blocks)])
    return {"summary": summary, "p": p}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["screen"])
    ap.add_argument("--blocks", type=int, default=60)
    ap.add_argument("--block-size", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    blocks = make_blocks(load_records(), args.blocks, args.block_size, args.seed)
    n_pairs = sum(len(b["pairs"]) for b in blocks)
    n_pos = sum(sum(b["gold"].values()) for b in blocks)
    print(f"{len(blocks)} blocks, {sum(len(b['records']) for b in blocks)} records, {n_pairs} cross-source pairs, "
          f"{n_pos} matches ({100 * n_pos / n_pairs:.0f}%)", flush=True)
    jev = Jev("jev-latest", args.concurrency)
    r = screen(jev, blocks, args.workers)
    print(f"{'method':38s} {'acc':>6s} {'prec':>6s} {'recall':>6s} {'F1':>6s} {'brier':>6s} {'auroc':>6s}")
    for name, m in r["summary"].items():
        print(f"{name:38s} {m['acc']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} "
              f"{m['brier']:6.3f} {m['auroc']:6.3f}")
    print(f"usage: {json.dumps(jev.usage())}")
    out = HERE / "results" / f"er_screen_b{args.blocks}_s{args.seed}.json"
    out.write_text(json.dumps({"args": vars(args), "blocks": blocks, **r, "usage": jev.usage()}, indent=1))
    print(f"saved {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
