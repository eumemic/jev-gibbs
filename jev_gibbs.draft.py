#!/usr/bin/env python3
"""Large Language Gibbs (Choi, Gouk & Whitammer 2026, §5.1) on TruthfulQA, with Jev as the judge.

Each claim is a binary variable (true / false answer to its question). A chain repeatedly
re-judges claims with Jev, showing it every other claim in the chunk with its current label.

Differences from the paper:
  * Jev returns P(true) for many claims in one call, so we update a *group* of claims per
    call: one claim from each question, so no claim's siblings are hidden while it is judged
    (a 4-colour block Gibbs sweep, 4 calls per sweep instead of 4m).
  * We keep Jev's probabilities and average them after burn-in (Rao-Blackwellised estimate)
    alongside the paper's majority vote over sampled labels.

Modes:
  probe     Cheap sanity check: does Jev's verdict move when the sibling labels change?
  zeroshot  Every claim judged alone, one call per chunk.
  gibbs     The chain.

Usage:
  export TYPESAFE_API_KEY=...
  python jev_gibbs.py probe
  python jev_gibbs.py zeroshot --chunks 2
  python jev_gibbs.py gibbs --chunks 2            # 2 x 16 questions, ~280 calls
  python jev_gibbs.py gibbs                       # full 256 questions, ~2,240 calls
  add --mock to any command to test the plumbing without calling the API.
Standard library only (Python 3.10+).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API_URL = "https://api.typesafe.ai/v1/systemone"
HERE = Path(__file__).resolve().parent
EPS = 1e-4

STATE_HEADER = (
    "Question-claim pairs. Each claim is a proposed answer to its question, "
    "followed by a verdict on whether the claim is a true, accurate answer."
)
ZEROSHOT_STATE = "A proposed answer to a question, to be judged for factual accuracy."
CRITERIA = {
    "true": "The claim is a factually true, accurate answer to the question",
    "false": "The claim is false, misleading, or a common misconception",
}


# ----------------------------------------------------------------------------- data

def load_data(path: Path):
    blob = json.loads(path.read_text())
    by_q: dict[int, list[dict]] = {}
    for uid, it in enumerate(blob["items"]):
        it = dict(it, uid=uid)
        by_q.setdefault(it["qid"], []).append(it)
    return [by_q[q] for q in blob["qids"]]  # list of questions, each a list of 4 claims


def chunk_questions(questions, m, n_chunks, seed):
    rng = random.Random(seed)
    qs = questions[:]
    rng.shuffle(qs)
    chunks = [qs[i:i + m] for i in range(0, len(qs), m)]
    return chunks[:n_chunks] if n_chunks else chunks


# ----------------------------------------------------------------------------- Jev client

class Jev:
    def __init__(self, model: str, mock: bool = False):
        self.model, self.mock = model, mock
        self.key = os.environ.get("TYPESAFE_API_KEY")
        if not mock and not self.key:
            sys.exit("Set TYPESAFE_API_KEY (or pass --mock).")
        self.calls = self.in_tok = self.out_tok = 0

    def nouls(self, state: str, claims: list[dict], labels: dict[int, bool] | None = None) -> dict[int, float]:
        """Return P(true) for each claim, judged against `state`."""
        if self.mock:
            return self._mock(claims, labels or {})
        questions = {
            f"c{c['uid']}": {
                "type": "noul",
                "instructions": f"Is this claim a true, accurate answer to its question?\n"
                                f"Question: {c['question']}\nClaim: {c['claim']}",
                "criteria": CRITERIA,
            }
            for c in claims
        }
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(2 ** attempt + random.random())
                    continue
                sys.exit(f"Jev API error {e.code}: {e.read().decode(errors='replace')[:500]}")
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 5:
                    time.sleep(2 ** attempt + random.random())
                    continue
                sys.exit(f"Jev API unreachable: {e}")
        self.calls += 1
        u = resp.get("usage", {})
        self.in_tok += u.get("input_tokens", 0)
        self.out_tok += u.get("output_tokens", 0)
        return {c["uid"]: float(resp["answers"][f"c{c['uid']}"]["noul"]) for c in claims}

    def _mock(self, claims, labels):
        # Fake judge: noisy view of the truth, nudged toward agreeing with sibling labels.
        self.calls += 1
        out = {}
        for c in claims:
            rng = random.Random(c["uid"] * 7919)
            base = (1.2 if c["label"] else -1.2) + rng.gauss(0, 1.5)
            sib = [labels[u] for u in labels if u in c.get("_sib_uids", ())]
            ctx = 0.3 * (sum(1 if s else -1 for s in sib))
            out[c["uid"]] = 1 / (1 + math.exp(-(base + ctx + random.gauss(0, 0.3))))
        return out


# ----------------------------------------------------------------------------- prompts

def render_state(questions, labels, hidden: set[int], rng) -> str:
    """Block-preserving shuffle: questions shuffled, claims shuffled within each question."""
    qs = questions[:]
    rng.shuffle(qs)
    blocks = []
    for q in qs:
        shown = [c for c in q if c["uid"] not in hidden]
        if not shown:
            continue
        rng.shuffle(shown)
        lines = [f"Question: {q[0]['question']}"]
        for c in shown:
            lines.append(f"Claim: {c['claim']}\nVerdict: {'True' if labels[c['uid']] else 'False'}")
        blocks.append("\n".join(lines))
    return STATE_HEADER + "\n\n" + "\n\n".join(blocks)


def temper(p: float, T: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    if T == 1:
        return p
    a, b = p ** (1 / T), (1 - p) ** (1 / T)
    return a / (a + b)


# ----------------------------------------------------------------------------- modes

def run_chain(jev: Jev, questions, args, seed: int):
    rng = random.Random(seed)
    claims = [c for q in questions for c in q]
    for q in questions:  # sibling links, used only by the mock judge
        for c in q:
            c["_sib_uids"] = {s["uid"] for s in q if s is not c}
    # Paper's init: half True, half False, shuffled.
    init = [True] * ((len(claims) + 1) // 2) + [False] * (len(claims) // 2)
    rng.shuffle(init)
    labels = {c["uid"]: v for c, v in zip(claims, init)}

    p_sum = {c["uid"]: 0.0 for c in claims}
    votes = {c["uid"]: 0 for c in claims}
    trace = []
    n_sweeps = args.burn_in + args.keep
    for sweep in range(n_sweeps):
        # Colour classes: reshuffle claim order within each question, class k = k-th claim of each.
        orders = [rng.sample(q, len(q)) for q in questions]
        classes = list(range(max(len(q) for q in questions)))
        rng.shuffle(classes)
        flips = 0
        for k in classes:
            group = [o[k] for o in orders if k < len(o)]
            hidden = {c["uid"] for c in group}
            state = render_state(questions, labels, hidden, rng)
            probs = jev.nouls(state, group, labels)
            for c in group:
                p = probs[c["uid"]]
                new = rng.random() < temper(p, args.temperature)
                flips += new != labels[c["uid"]]
                labels[c["uid"]] = new
                if sweep >= args.burn_in:
                    p_sum[c["uid"]] += p
        if sweep >= args.burn_in:
            for u, v in labels.items():
                votes[u] += v
        acc_now = sum(labels[c["uid"]] == bool(c["label"]) for c in claims) / len(claims)
        trace.append({"sweep": sweep, "flips": flips, "acc_current_labels": round(acc_now, 4)})
        if args.verbose:
            print(f"  [chain {seed}] sweep {sweep:3d} flips={flips:3d} acc={acc_now:.3f}", flush=True)
    return {
        c["uid"]: {
            "label": c["label"],
            "p_rb": p_sum[c["uid"]] / args.keep,
            "vote_frac": votes[c["uid"]] / args.keep,
        }
        for c in claims
    }, trace


def run_zeroshot(jev: Jev, questions):
    claims = [c for q in questions for c in q]
    probs = jev.nouls(ZEROSHOT_STATE, claims)
    return {c["uid"]: {"label": c["label"], "p": probs[c["uid"]]} for c in claims}


def run_probe(jev: Jev, questions, n: int, seed: int):
    """For n claims: P(true) given siblings labelled with the truth vs. with flipped labels."""
    rng = random.Random(seed)
    rows = []
    for q in rng.sample(questions, n):
        target = rng.choice(q)
        sibs = [c for c in q if c is not target]
        for name, flip in (("truth", False), ("flipped", True)):
            labels = {c["uid"]: bool(c["label"]) ^ flip for c in sibs}
            p = jev.nouls(render_state([sibs], labels, set(), rng), [target], labels)[target["uid"]]
            rows.append({"uid": target["uid"], "label": target["label"], "context": name, "p": p})
        p_alone = jev.nouls(ZEROSHOT_STATE, [target])[target["uid"]]
        rows.append({"uid": target["uid"], "label": target["label"], "context": "none", "p": p_alone})
    return rows


# ----------------------------------------------------------------------------- metrics

def metrics(ps: list[float], ys: list[int]) -> dict:
    n = len(ps)
    acc = sum((p > 0.5) == bool(y) for p, y in zip(ps, ys)) / n
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
    bins = [[] for _ in range(10)]
    for p, y in zip(ps, ys):
        bins[min(int(p * 10), 9)].append((p, y))
    ece = sum(len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b))
              for b in bins if b)
    conf = [(abs(2 * p - 1), (p > 0.5) == bool(y)) for p, y in zip(ps, ys)]
    conf.sort(reverse=True)
    sel = {}
    for frac in (0.25, 0.5, 0.75):
        top = conf[: max(1, int(frac * n))]
        sel[f"acc_top{int(frac * 100)}pct_confident"] = round(sum(ok for _, ok in top) / len(top), 4)
    return {"n": n, "accuracy": round(acc, 4), "brier": round(brier, 4), "ece": round(ece, 4), **sel}


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["probe", "zeroshot", "gibbs"])
    ap.add_argument("--data", type=Path, default=HERE / "truthfulqa_eval.json")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--m", type=int, default=16, help="questions per chunk (paper's best: 16 -> 64 claims)")
    ap.add_argument("--chunks", type=int, default=0, help="number of chunks to run (0 = all 256/m)")
    ap.add_argument("--burn-in", type=int, default=10, help="sweeps discarded (paper: 10n steps = 10 sweeps)")
    ap.add_argument("--keep", type=int, default=25, help="sweeps averaged after burn-in (paper: 25 snapshots)")
    ap.add_argument("--temperature", type=float, default=1.0, help=">1 softens Jev's probabilities before sampling")
    ap.add_argument("--probe-n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16, help="chunks run concurrently")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="fake judge, no API calls")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    questions = load_data(args.data)
    jev = Jev(args.model, mock=args.mock)
    t0 = time.time()
    out_path = args.out or HERE / "results" / f"{args.mode}{'_mock' if args.mock else ''}_seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict = {"mode": args.mode, "args": {k: str(v) for k, v in vars(args).items()}}

    if args.mode == "probe":
        rows = run_probe(jev, questions, args.probe_n, args.seed)
        by = {}
        for r in rows:
            by.setdefault(r["uid"], {})[r["context"]] = r["p"]
        shift = [abs(v["truth"] - v["flipped"]) for v in by.values()]
        print(f"\nProbe over {len(by)} claims (P(true) with siblings labelled correctly vs. flipped vs. alone):")
        for uid, v in by.items():
            lab = next(r["label"] for r in rows if r["uid"] == uid)
            print(f"  claim {uid:4d} (truth={'T' if lab else 'F'})  correct-ctx={v['truth']:.2f}  "
                  f"flipped-ctx={v['flipped']:.2f}  alone={v['none']:.2f}")
        mean_shift = sum(shift) / len(shift)
        print(f"\nMean |P(correct ctx) - P(flipped ctx)| = {mean_shift:.3f}")
        print("Rough read: < 0.05 means Jev mostly ignores the other labels and the Gibbs loop has nothing to work "
              "with; > 0.15 means context clearly moves it.")
        result.update(rows=rows, mean_context_shift=mean_shift)

    else:
        chunks = chunk_questions(questions, args.m, args.chunks, args.seed)
        if args.mode == "zeroshot":
            with ThreadPoolExecutor(args.workers) as ex:
                parts = list(ex.map(lambda ch: run_zeroshot(jev, ch), chunks))
            per = {u: v for p in parts for u, v in p.items()}
            ps, ys = [v["p"] for v in per.values()], [v["label"] for v in per.values()]
            result.update(metrics=metrics(ps, ys), per_claim=per)
            print(f"\nZero-shot on {len(per)} claims:", json.dumps(result["metrics"], indent=1))
        else:
            n_calls = len(chunks) * (args.burn_in + args.keep) * 4
            print(f"{len(chunks)} chunk(s) x {args.m} questions; ~{n_calls} Jev calls "
                  f"(each with ~{args.m * 4 - args.m} labelled claims of context).", flush=True)
            with ThreadPoolExecutor(args.workers) as ex:
                parts = list(ex.map(lambda ich: run_chain(jev, ich[1], args, args.seed * 1000 + ich[0]),
                                    enumerate(chunks)))
            per = {u: v for p, _ in parts for u, v in p.items()}
            ys = [v["label"] for v in per.values()]
            m_rb = metrics([v["p_rb"] for v in per.values()], ys)
            m_vote = metrics([v["vote_frac"] for v in per.values()], ys)
            result.update(metrics_rao_blackwell=m_rb, metrics_majority_vote=m_vote, per_claim=per,
                          traces=[t for _, t in parts])
            print(f"\nGibbs on {len(per)} claims")
            print("  averaged Jev probabilities:", json.dumps(m_rb))
            print("  majority vote (paper's way):", json.dumps(m_vote))

    result["usage"] = {"calls": jev.calls, "input_tokens": jev.in_tok, "output_tokens": jev.out_tok,
                       "seconds": round(time.time() - t0, 1)}
    print(f"\n{jev.calls} calls, {jev.in_tok:,} input / {jev.out_tok:,} output tokens, "
          f"{result['usage']['seconds']}s. Saved {out_path}")
    out_path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
