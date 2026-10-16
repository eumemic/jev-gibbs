#!/usr/bin/env python3
"""Summarise results/*.json into one table, and compare methods claim by claim."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_gibbs import load_data, metrics, mean_std  # noqa: E402

HERE = Path(__file__).resolve().parent
R = HERE / "results"

# Paper Table 1 / Table 2, TruthfulQA column (mean over 5 seeds).
PAPER = {
    "Llama-3.1-8B": {"zero-shot": 0.588, "Gibbs m=1": 0.708, "Gibbs m=4": 0.715, "Gibbs m=16": 0.744},
    "OLMo-3-32B": {"zero-shot": 0.610, "Gibbs m=1": 0.664, "Gibbs m=4": 0.693, "Gibbs m=16": 0.776},
    "Llama-3.1-8B-Instruct": {"zero-shot": 0.658, "Gibbs m=1": 0.771, "Gibbs m=4": 0.771, "Gibbs m=16": 0.793},
    "OLMo-3.1-32B-Instruct": {"zero-shot": 0.755, "Gibbs m=1": 0.803, "Gibbs m=4": 0.811, "Gibbs m=16": 0.835},
}


def avg(ms) -> dict:
    ms = list(ms)
    return {k: sum(m[k] for m in ms) / len(ms) for k in ("brier", "ece", "auroc", "acc@50%")}


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    by_q = load_data(HERE / "truthfulqa_eval.json")
    claims = [c for q in by_q.values() for c in q]
    ys = [c["label"] for c in claims]
    rows = []  # (name, per-seed accuracies, metrics on seed 0 or pooled, calls, usd)

    zs = load("zeroshot")
    if zs:
        p = {int(k): v for k, v in zs["criteria"]["p"].items()}
        m = metrics([p[c["uid"]] for c in claims], ys)
        rows.append(("Jev zero-shot", [m["acc"]], m, 16, zs["usage"]["usd"] / 2))
    for m_ in (1, 4, 16):
        for suffix, desc in (("", "1 pass"), ("_x5", "5 passes")):
            ctx = load(f"context_m{m_}{suffix}")
            if ctx:
                sm = ctx["seed_metrics"]
                rows.append((f"Jev context, no verdicts, m={m_}, {desc}", [s["acc"] for s in sm.values()],
                             avg(sm.values()), ctx["usage"]["calls"] / len(sm), ctx["usage"]["usd"] / len(sm)))
    for m_ in (1, 4, 16):
        for sched in ("block", "seq"):
            for tag in ("", "_hint", "_zsinit"):
                g = load(f"gibbs_m{m_}_{sched}{tag}")
                if not g:
                    continue
                sm = g["seed_metrics"]
                for est in ("vote", "rb"):
                    label = f"Jev Gibbs m={m_} {sched}{tag.replace('_', ' ')} [{est}]"
                    rows.append((label, [s[est]["acc"] for s in sm.values()], avg(s[est] for s in sm.values()),
                                 g["usage"]["calls"] / len(sm), g["usage"]["usd"] / len(sm)))

    print(f"{'method':48s} {'acc (mean ± sd, 5 seeds)':>27s} {'brier':>6s} {'ece':>6s} {'auroc':>6s} "
          f"{'acc@50%':>7s} {'calls':>7s} {'$':>6s}")
    for name, accs, m, calls, usd in rows:
        acc = mean_std(accs) if len(accs) > 1 else f"{accs[0]:.3f}"
        print(f"{name:48s} {acc:>27s} {m['brier']:6.3f} {m['ece']:6.3f} {m['auroc']:6.3f} {m['acc@50%']:7.3f} "
              f"{calls:7.0f} {usd:6.2f}")
    print("\nPaper, TruthfulQA (calls/run not comparable: one token-level LM call per Gibbs step)")
    for model, d in PAPER.items():
        print(f"  {model:24s} " + "  ".join(f"{k} {v:.3f}" for k, v in d.items()))

    # Same partitions per seed, so Gibbs and the no-verdict control can be compared seed by seed.
    for suffix, desc in (("", "1 pass"), ("_x5", "5 passes averaged")):
        print(f"\nGibbs minus no-verdict control ({desc}), per seed, accuracy points:")
        for m_ in (1, 4, 16):
            ctx = load(f"context_m{m_}{suffix}")
            for sched in ("block", "seq"):
                g = load(f"gibbs_m{m_}_{sched}")
                if not (g and ctx):
                    continue
                for est in ("vote", "rb"):
                    d = [100 * (g["seed_metrics"][s][est]["acc"] - ctx["seed_metrics"][s]["acc"])
                         for s in g["seed_metrics"]]
                    print(f"  m={m_:2d} {sched:5s} [{est:4s}]: " + " ".join(f"{x:+.1f}" for x in d)
                          + f"   mean {sum(d) / len(d):+.2f}")

    # Claim-level comparison on seed 0: where do Gibbs and the no-verdict control disagree with zero-shot?
    g, ctx = load("gibbs_m16_block"), load("context_m16")
    if zs and g and ctx:
        z = {int(k): v > 0.5 for k, v in zs["criteria"]["p"].items()}
        gv = {int(k): v["p_rb"] > 0.5 for k, v in g["per_seed"]["0"].items()}
        cv = {int(k): v > 0.5 for k, v in ctx["per_seed"]["0"].items()}
        y = {c["uid"]: bool(c["label"]) for c in claims}

        def changes(a, b):
            fixed = sum(a[u] != y[u] and b[u] == y[u] for u in y)
            broke = sum(a[u] == y[u] and b[u] != y[u] for u in y)
            return f"fixes {fixed}, breaks {broke}"
        print("\nSeed 0, m=16, claim by claim:")
        print(f"  zero-shot -> context(no verdicts): {changes(z, cv)}")
        print(f"  zero-shot -> Gibbs (rb):           {changes(z, gv)}")
        print(f"  context   -> Gibbs (rb):           {changes(cv, gv)}")
        print(f"  Gibbs and context agree on {sum(gv[u] == cv[u] for u in y)}/{len(y)} claims")


if __name__ == "__main__":
    main()
