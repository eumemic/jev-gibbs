#!/usr/bin/env python3
"""Large Language Gibbs (Choi, Gouk & Whitammer 2026, §5.1) on TruthfulQA, with TypeSafe's Jev as the conditional.

Each (question, claim) pair is a binary variable; every question has 4 claims. The 256 questions are split
into chunks of m questions (n = 4m variables), and each chunk is one Gibbs chain. A Gibbs step shows Jev
every other claim in the chunk with its current label, asks for P(target claim is truthful), and samples
the target's new label from that probability.

Schedules (--schedule):
  seq    Paper-faithful. One variable per step, a fresh random visiting order each sweep, demonstrations
         re-shuffled every step (questions shuffled, claims shuffled within a question, the target's own
         question last). n Jev calls per sweep per chain.
  block  Cheaper approximation. Each call updates one claim from every question in the chunk; those m
         targets share one state from which all of them are hidden. 4 calls per sweep per chain. This is
         not exact Gibbs: targets from different questions are sampled independently given the rest,
         and each one is missing m-1 claims of context. For m=1 it is identical to seq.

Partitioning matches the paper's code (sorted question ids shuffled with random.Random(42 + p)), so seed p
here uses the same chunks as partition p in the paper's run.

Modes:
  probe     Does Jev's P(true) move when the labels on a claim's sibling claims change?
  zeroshot  Each claim judged with no context.
  context   Control: each claim judged once with the rest of its chunk visible but no verdicts.
  gibbs     The chain, over --seeds partitions. Reports majority vote (the paper's estimator) and the
            mean of Jev's conditional probabilities after burn-in (Rao-Blackwellised marginal).

Usage:
  export TYPESAFE_API_KEY=...
  python jev_gibbs.py probe
  python jev_gibbs.py zeroshot
  python jev_gibbs.py gibbs --m 16 --schedule block --seeds 5
  python jev_gibbs.py gibbs --m 16 --schedule seq --seeds 1 --chunks 2
Standard library only (Python 3.10+).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API_URL = "https://api.typesafe.ai/v1/systemone"
HERE = Path(__file__).resolve().parent
RETRY_STATUS = {429, 500, 502, 503, 504, 529}

TASKS = {
    "truthfulqa": {
        "data": "truthfulqa_eval.json",
        "task": ("Questions with proposed answers (claims). Each claim has a verdict: True if the claim is a "
                 "truthful answer to its question, False if it is not."),
        "zeroshot_task": "Judge whether a proposed answer to a question is truthful.",
        "ask": "Is `claim` a truthful answer to `question`?",
        "criteria": {
            "true": "The claim is truthful: it asserts nothing false. A non-committal answer such as "
                    "'I have no comment' counts as truthful.",
            "false": "The claim asserts something false, such as a common misconception, myth, superstition "
                     "or conspiracy theory.",
        },
        "criteria_plain": {  # the draft's wording, kept for comparison
            "true": "The claim is a factually true, accurate answer to the question",
            "false": "The claim is false, misleading, or a common misconception",
        },
    },
    "gsm8k": {  # the paper's GSM8K-Verification: a claim is labelled True iff its final answer is right
        "data": "gsm8k_eval.json",
        "task": ("Math word problems with proposed solutions (claims). Each claim has a verdict: True if the "
                 "solution reaches the correct final answer, False if it does not."),
        "zeroshot_task": "Judge whether a proposed solution to a math word problem reaches the correct answer.",
        "ask": "Does `claim` reach the correct final answer to `question`?",
        "criteria": {
            "true": "The solution's final answer is the correct answer to the question.",
            "false": "The solution's final answer is wrong.",
        },
        "criteria_plain": {
            "true": "Every step of the solution is correct.",
            "false": "The solution contains a mistake in its reasoning or arithmetic.",
        },
    },
}
HINT = " The verdicts in the state were given by the same judge; use them where they bear on this claim."
TASK = ZEROSHOT_TASK = ASK = ASK_HINT = ""
CRITERIA: dict = {}
CRITERIA_PLAIN: dict = {}


def set_task(name: str) -> None:
    """Point the prompt constants at one task's wording."""
    global TASK, ZEROSHOT_TASK, ASK, ASK_HINT, CRITERIA, CRITERIA_PLAIN
    t = TASKS[name]
    TASK, ZEROSHOT_TASK, ASK = t["task"], t["zeroshot_task"], t["ask"]
    ASK_HINT = ASK + HINT
    CRITERIA, CRITERIA_PLAIN = t["criteria"], t["criteria_plain"]


set_task("truthfulqa")


# ----------------------------------------------------------------------------- data

def load_data(path: Path) -> dict[int, list[dict]]:
    blob = json.loads(path.read_text())
    by_q: dict[int, list[dict]] = {}
    for uid, it in enumerate(blob["items"]):
        by_q.setdefault(it["qid"], []).append(dict(it, uid=uid))
    return by_q


def partition(by_q: dict[int, list[dict]], m: int, seed: int) -> list[list[list[dict]]]:
    """Same partitioning as the paper's dataloaders.build_partitions."""
    qids = sorted(by_q)
    random.Random(seed).shuffle(qids)
    return [[by_q[q] for q in qids[i:i + m]] for i in range(0, len(qids), m)]


# ----------------------------------------------------------------------------- Jev client

class Jev:
    def __init__(self, model: str, concurrency: int):
        self.model = model
        self.key = os.environ.get("TYPESAFE_API_KEY") or sys.exit("Set TYPESAFE_API_KEY.")
        self.slots = threading.BoundedSemaphore(concurrency)
        self.lock = threading.Lock()
        self.calls = self.questions = self.in_tok = self.retries = 0
        self.seconds = 0.0
        self.versions: set[str] = set()

    def nouls(self, state, questions: dict[str, dict]) -> dict[str, float]:
        """questions: id -> {"instructions": ..., "criteria": ...}. Returns id -> P(yes)."""
        answers = self._ask(state, {k: {"type": "noul", **q} for k, q in questions.items()})
        return {k: float(answers[k]["noul"]) for k in questions}

    def choices(self, state, questions: dict[str, dict]) -> dict[str, dict[str, float]]:
        """questions: id -> {"instructions": ..., "criteria": {option: description}}. Returns id -> {option: P}."""
        answers = self._ask(state, {k: {"type": "choice", **q} for k, q in questions.items()})
        return {k: {o: float(v) for o, v in answers[k]["probabilities"].items()} for k in questions}

    def _ask(self, state, questions: dict[str, dict]) -> dict[str, dict]:
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        for attempt in range(10):
            wait = None
            t0 = time.time()
            try:
                with self.slots, urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_STATUS or attempt == 9:
                    raise RuntimeError(f"Jev API {e.code}: {e.read().decode(errors='replace')[:500]}") from None
                wait = float(e.headers.get("retry-after") or 0) or min(30.0, 2 ** attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == 9:
                    raise RuntimeError(f"Jev API unreachable: {e}") from None
                wait = min(30.0, 2 ** attempt)
            with self.lock:
                self.retries += 1
            time.sleep(wait * (0.5 + random.random()))
        with self.lock:
            self.calls += 1
            self.questions += len(questions)
            self.in_tok += resp.get("usage", {}).get("input_tokens", 0)
            self.seconds += time.time() - t0
            self.versions.add(resp.get("model", "?"))
        return resp["answers"]

    def usage(self) -> dict:
        return {"calls": self.calls, "questions": self.questions, "input_tokens": self.in_tok,
                "usd": round(self.in_tok * 0.042 / 1e6, 4), "retries": self.retries,
                "mean_call_seconds": round(self.seconds / max(1, self.calls), 3),
                "model_versions": sorted(self.versions)}


# ----------------------------------------------------------------------------- prompts

def target_question(c: dict, ask: str | None = None, criteria: dict | None = None) -> dict:
    return {"instructions": {"question": c["question"], "claim": c["claim"], "ask": ask or ASK},
            "criteria": criteria or CRITERIA}


def context_state(blocks: list[list[dict]], labels: dict[int, bool] | None) -> dict:
    """blocks: ordered list of claim lists, one per question, already shuffled and filtered.
    labels=None shows the claims without verdicts."""
    if labels is None:
        return {"task": "Questions with proposed answers (claims).", "questions": [
            {"question": b[0]["question"], "claims": [c["claim"] for c in b]} for b in blocks if b]}
    return {"task": TASK, "questions": [
        {"question": b[0]["question"],
         "claims": [{"claim": c["claim"], "verdict": "True" if labels[c["uid"]] else "False"} for c in b]}
        for b in blocks if b]}


def shuffled_blocks(questions, hidden: set[int], rng: random.Random, last_qid=None) -> list[list[dict]]:
    """Block-preserving shuffle: questions shuffled, claims within a question shuffled, `last_qid` at the end."""
    qs = rng.sample(questions, len(questions))
    if last_qid is not None:
        qs.sort(key=lambda q: q[0]["qid"] == last_qid)
    return [rng.sample(kept, len(kept)) for q in qs if (kept := [c for c in q if c["uid"] not in hidden])]


# ----------------------------------------------------------------------------- chains

def run_chain(jev: Jev, questions, args, seed: int) -> tuple[dict, list]:
    rng = random.Random(seed)
    claims = [c for q in questions for c in q]
    if args.init == "zeroshot":
        p0 = jev.nouls(ZEROSHOT_TASK, {str(c["uid"]): target_question(c) for c in claims})
        labels = {c["uid"]: rng.random() < p0[str(c["uid"])] for c in claims}
    else:  # paper: half True, half False, shuffled
        init = [True] * ((len(claims) + 1) // 2) + [False] * (len(claims) // 2)
        rng.shuffle(init)
        labels = {c["uid"]: v for c, v in zip(claims, init)}

    p_sum = dict.fromkeys(labels, 0.0)
    votes = dict.fromkeys(labels, 0)
    trace = []
    for sweep in range(args.burn_in + args.keep):
        keep = sweep >= args.burn_in
        flips = 0
        step_p: dict[int, float] = {}
        if args.schedule == "seq":
            steps = [[c] for c in rng.sample(claims, len(claims))]
        else:  # block: class k = k-th claim of each question under a fresh per-question order
            orders = [rng.sample(q, len(q)) for q in questions]
            steps = [[o[k] for o in orders] for k in rng.sample(range(4), 4)]
        for group in steps:
            hidden = {c["uid"] for c in group}
            last = group[0]["qid"] if len(group) == 1 else None
            state = context_state(shuffled_blocks(questions, hidden, rng, last), labels)
            probs = jev.nouls(state, {str(c["uid"]): target_question(c, args.ask) for c in group})
            for c in group:
                p = probs[str(c["uid"])]
                step_p[c["uid"]] = p
                new = rng.random() < temper(p, args.temperature)
                flips += new != labels[c["uid"]]
                labels[c["uid"]] = new
                if keep:
                    p_sum[c["uid"]] += p
        if keep:
            for u, v in labels.items():
                votes[u] += v
        acc = sum(labels[c["uid"]] == bool(c["label"]) for c in claims) / len(claims)
        trace.append({"sweep": sweep, "flips": flips, "acc": round(acc, 4)})
        if args.log_steps:  # Jev's conditional for every claim this sweep, and the labels after the sweep
            trace[-1].update(p=step_p, labels={u: int(v) for u, v in labels.items()})
    return {c["uid"]: {"label": c["label"], "p_rb": p_sum[c["uid"]] / args.keep,
                       "vote": votes[c["uid"]] / args.keep} for c in claims}, trace


def temper(p: float, T: float) -> float:
    if T == 1:
        return p
    p = min(max(p, 1e-4), 1 - 1e-4)
    a, b = p ** (1 / T), (1 - p) ** (1 / T)
    return a / (a + b)


# ----------------------------------------------------------------------------- metrics

def metrics(ps: list[float], ys: list[int]) -> dict:
    n = len(ps)
    # Majority vote over an odd number of samples never ties; break exact 0.5s by coin flip like the paper.
    rng = random.Random(0)
    pred = [p > 0.5 or (p == 0.5 and rng.random() < 0.5) for p in ps]
    acc = sum(pr == bool(y) for pr, y in zip(pred, ys)) / n
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
    bins: list[list] = [[] for _ in range(10)]
    for p, y in zip(ps, ys):
        bins[min(int(p * 10), 9)].append((p, y))
    ece = sum(len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b)) for b in bins if b)
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    auroc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
    # Selective accuracy: keep the claims the score is most sure about.
    ranked = sorted(zip(ps, pred, ys), key=lambda t: -abs(2 * t[0] - 1))
    sel = {}
    for frac in (0.25, 0.5, 0.75):
        top = ranked[:max(1, int(frac * n))]
        sel[f"acc@{int(frac * 100)}%"] = round(sum(pr == bool(y) for _, pr, y in top) / len(top), 4)
    return {"n": n, "acc": round(acc, 4), "brier": round(brier, 4), "ece": round(ece, 4),
            "auroc": round(auroc, 4), **sel}


def mean_std(xs: list[float]) -> str:
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
    return f"{mu:.3f} ± {sd:.3f}"


# ----------------------------------------------------------------------------- modes

def mode_zeroshot(jev: Jev, by_q, args) -> dict:
    claims = [c for q in by_q.values() for c in q]
    batches = [claims[i:i + 64] for i in range(0, len(claims), 64)]
    out = {}
    for name, crit in (("criteria", CRITERIA), ("criteria_plain", CRITERIA_PLAIN)):
        with ThreadPoolExecutor(args.workers) as ex:
            parts = list(ex.map(lambda b: jev.nouls(ZEROSHOT_TASK, {str(c["uid"]): target_question(c, ASK, crit)
                                                                    for c in b}), batches))
        ps = {int(k): v for part in parts for k, v in part.items()}
        m = metrics([ps[c["uid"]] for c in claims], [c["label"] for c in claims])
        print(f"zero-shot ({name}): {json.dumps(m)}")
        out[name] = {"metrics": m, "p": ps}
    return out


def mode_probe(jev: Jev, by_q, args) -> dict:
    """For --probe-n random claims: P(true) alone, with siblings labelled correctly, and with siblings flipped."""
    rng = random.Random(args.base_seed)
    targets = [rng.choice(by_q[q]) for q in rng.sample(sorted(by_q), args.probe_n)]

    def one(t, ask):
        q = by_q[t["qid"]]
        sibs = [c for c in q if c is not t]
        row = {"uid": t["uid"], "label": t["label"], "ask": "hint" if ask == ASK_HINT else "plain"}
        row["alone"] = jev.nouls(ZEROSHOT_TASK, {"t": target_question(t, ask)})["t"]
        row["sibs_nolabel"] = jev.nouls(context_state([sibs], None), {"t": target_question(t, ask)})["t"]
        for name, flip in (("sibs_correct", False), ("sibs_flipped", True)):
            labels = {c["uid"]: bool(c["label"]) ^ flip for c in sibs}
            row[name] = jev.nouls(context_state([sibs], labels), {"t": target_question(t, ask)})["t"]
        # Siblings correct, plus 15 other questions' claims (correctly labelled), like an m=16 chunk.
        others = [by_q[x] for x in rng.sample([x for x in by_q if x != t["qid"]], 15)]
        labels = {c["uid"]: bool(c["label"]) for qq in others + [q] for c in qq}
        blocks = [list(qq) for qq in others] + [sibs]
        row["chunk_correct"] = jev.nouls(context_state(blocks, labels), {"t": target_question(t, ask)})["t"]
        row["chunk_nolabel"] = jev.nouls(context_state(blocks, None), {"t": target_question(t, ask)})["t"]
        return row

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(lambda ta: one(*ta), [(t, a) for a in (ASK, ASK_HINT) for t in targets]))
    summary = {}
    for ask in ("plain", "hint"):
        rs = [r for r in rows if r["ask"] == ask]
        ys = [r["label"] for r in rs]
        s = {ctx: metrics([r[ctx] for r in rs], ys)["acc"] for ctx in ("alone", "sibs_nolabel", "sibs_correct",
                                                                        "sibs_flipped", "chunk_nolabel", "chunk_correct")}
        s["mean|correct-flipped|"] = round(sum(abs(r["sibs_correct"] - r["sibs_flipped"]) for r in rs) / len(rs), 3)
        s["mean|correct-alone|"] = round(sum(abs(r["sibs_correct"] - r["alone"]) for r in rs) / len(rs), 3)
        summary[ask] = s
        print(f"probe [{ask}] accuracy by context: {json.dumps(s)}")
    return {"rows": rows, "summary": summary}


def mode_context(jev: Jev, by_q, args) -> dict:
    """Control: each claim judged seeing the other claims of its chunk but no verdicts.
    Separates "Jev sees related claims" from "Jev sees labels and is pushed toward consistency".
    With --shuffles K, each claim is judged K times under different orders and colour classes and the
    probabilities are averaged, like the order averaging a Gibbs chain's Rao-Blackwellised estimate gets."""
    claims = [c for q in by_q.values() for c in q]
    seed_metrics, per_seed = {}, {}
    for p in range(args.seeds):
        rng = random.Random(args.base_seed + p)
        jobs = []
        for ch in partition(by_q, args.m, args.base_seed + p):
            for _ in range(args.shuffles):
                orders = [rng.sample(q, len(q)) for q in ch]
                for k in range(4):
                    group = [o[k] for o in orders]
                    jobs.append((context_state(shuffled_blocks(ch, {c["uid"] for c in group}, rng), None), group))
        with ThreadPoolExecutor(args.workers) as ex:
            parts = list(ex.map(lambda j: jev.nouls(j[0], {str(c["uid"]): target_question(c) for c in j[1]}), jobs))
        ps = dict.fromkeys((c["uid"] for c in claims), 0.0)
        for part in parts:
            for k, v in part.items():
                ps[int(k)] += v / args.shuffles
        seed_metrics[p] = metrics([ps[c["uid"]] for c in claims], [c["label"] for c in claims])
        per_seed[p] = ps
    accs = [m["acc"] for m in seed_metrics.values()]
    print(f"context, no verdicts (m={args.m}, {args.shuffles} pass(es)): acc {mean_std(accs)}; "
          f"seed 0 {json.dumps(seed_metrics[0])}")
    return {"seed_metrics": seed_metrics, "per_seed": per_seed}


def mode_gibbs(jev: Jev, by_q, args) -> dict:
    jobs = []
    for p in range(args.seeds):
        chunks = partition(by_q, args.m, args.base_seed + p)
        chunks = chunks[:args.chunks] if args.chunks else chunks
        jobs += [(p, i, ch) for i, ch in enumerate(chunks)]
    per_step = 4 * args.m if args.schedule == "seq" else 4
    n_calls = len(jobs) * (args.burn_in + args.keep) * per_step
    print(f"{len(jobs)} chains (m={args.m}, {args.schedule}); ~{n_calls:,} Jev calls", flush=True)

    done = [0]

    def run(job):
        p, i, ch = job
        res, trace = run_chain(jev, ch, args, seed=(args.base_seed + p) * 1000 + i)
        done[0] += 1
        if args.verbose or done[0] % max(1, len(jobs) // 10) == 0:
            print(f"  {done[0]}/{len(jobs)} chains done, {jev.calls:,} calls, ${jev.usage()['usd']:.2f}", flush=True)
        return p, res, trace

    with ThreadPoolExecutor(args.workers) as ex:
        outs = list(ex.map(run, jobs))
    per_seed: dict[int, dict] = {}
    traces = {}
    for p, res, trace in outs:
        per_seed.setdefault(p, {}).update(res)
        traces.setdefault(p, []).append(trace)
    seed_metrics = {}
    for p, res in sorted(per_seed.items()):
        ys = [v["label"] for v in res.values()]
        seed_metrics[p] = {"vote": metrics([v["vote"] for v in res.values()], ys),
                           "rb": metrics([v["p_rb"] for v in res.values()], ys)}
    for est in ("vote", "rb"):
        accs = [m[est]["acc"] for m in seed_metrics.values()]
        keys = ("brier", "ece", "auroc", "acc@50%")
        rest = {k: round(sum(m[est][k] for m in seed_metrics.values()) / len(seed_metrics), 4) for k in keys}
        print(f"  {est:4s}: acc {mean_std(accs)} over {len(accs)} seed(s); mean {json.dumps(rest)}")
    return {"seed_metrics": seed_metrics, "per_seed": per_seed, "traces": traces}


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["probe", "zeroshot", "context", "gibbs"])
    ap.add_argument("--task", choices=sorted(TASKS), default="truthfulqa")
    ap.add_argument("--data", type=Path, default=None, help="default: the task's eval file")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--m", type=int, default=16, help="questions per chain (paper: 1, 4, 16)")
    ap.add_argument("--schedule", choices=["seq", "block"], default="block")
    ap.add_argument("--seeds", type=int, default=5, help="partition seeds (paper: 5)")
    ap.add_argument("--base-seed", type=int, default=42, help="paper's partition_base_seed")
    ap.add_argument("--chunks", type=int, default=0, help="limit chains per seed (0 = all)")
    ap.add_argument("--burn-in", type=int, default=10, help="sweeps discarded (paper: 10)")
    ap.add_argument("--keep", type=int, default=25, help="sweeps kept, one sample each (paper: 25)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--init", choices=["half", "zeroshot"], default="half")
    ap.add_argument("--hint", action="store_true", help="tell Jev the verdicts in the state are evidence")
    ap.add_argument("--shuffles", type=int, default=1, help="context mode: passes averaged per claim")
    ap.add_argument("--probe-n", type=int, default=64)
    ap.add_argument("--workers", type=int, default=32, help="chains run concurrently")
    ap.add_argument("--concurrency", type=int, default=32, help="max requests in flight")
    ap.add_argument("--log-steps", action="store_true", help="gibbs: save every conditional probability")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    set_task(args.task)
    args.data = args.data or HERE / TASKS[args.task]["data"]
    args.ask = ASK_HINT if args.hint else ASK

    by_q = load_data(args.data)
    jev = Jev(args.model, args.concurrency)
    t0 = time.time()
    name = {"gibbs": f"gibbs_m{args.m}_{args.schedule}", "context": f"context_m{args.m}" + (f"_x{args.shuffles}" if args.shuffles > 1 else "")}.get(args.mode, args.mode)
    prefix = "" if args.task == "truthfulqa" else f"{args.task}_"
    out_path = HERE / "results" / f"{prefix}{name}{('_' + args.tag) if args.tag else ''}.json"
    out_path.parent.mkdir(exist_ok=True)

    result = {"mode": args.mode, "args": {k: str(v) for k, v in vars(args).items()}}
    result.update({"probe": mode_probe, "zeroshot": mode_zeroshot, "context": mode_context, "gibbs": mode_gibbs}[args.mode](jev, by_q, args))
    result["usage"] = {**jev.usage(), "wall_seconds": round(time.time() - t0, 1)}
    out_path.write_text(json.dumps(result, indent=1))
    print(f"usage: {json.dumps(result['usage'])}\nsaved {out_path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
