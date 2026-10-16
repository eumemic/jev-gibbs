#!/usr/bin/env python3
"""Calibration of the kinship labels' confidences, per method, from the `kin.py compare` caches.

Each method gives a probability vector over the 23 kinship terms for every variable. The label is the top
term and its confidence is that term's probability: Jev's own for one pass and one refinement step, the
averaged conditionals for Gibbs [rb], and the fraction of kept sweeps for Gibbs [vote].

Reported per method:
  acc        accuracy of the top label
  ECE        top-label expected calibration error (10 equal-width bins)
  over       mean confidence minus accuracy (positive = overconfident)
  Brier      multiclass Brier score
  NLL        negative log probability of the gold term (floored at 1e-3)
  AUROC      how well the confidence separates right from wrong labels
  keep@95    share of labels kept when keeping the most confident ones while their accuracy stays >= 95%
  mass_ok    share of total confidence that sits on correct labels
Uncertainty: 90% intervals from 1,000 bootstrap resamples of stories, for each method's difference from one pass.

  python3 calib.py           the table above, per method
  python3 calib.py --fixes   chain labels with confidence from another source, or rescaled on labelled stories
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = [("one_pass", "p"), ("refine1", "p"), ("gibbs_T1", "rb"), ("gibbs_T1", "vote"),
           ("gibbs_T0.3", "rb"), ("gibbs_T0.3", "vote"), ("gibbs_T0.1", "rb"), ("gibbs_T0.1", "vote")]


def load(k: int) -> tuple[list[dict], dict]:
    stories = json.loads((HERE / "results/kin_compare_k4_6.json").read_text())["results"][str(k)]["stories"]
    recs = {}
    for line in (HERE / f"results/kin_compare_k{k}_s0_b5_n20.jsonl").read_text().splitlines():
        x = json.loads(line)
        recs[(x["method"], x["si"])] = x
    return stories, recs


def items(stories, recs, method, est):
    """One (story index, confidence, correct, probability vector, gold) per variable."""
    out = []
    for si, st in enumerate(stories):
        p = recs[(method, si)][est]
        for v, gold in st["gold"].items():
            top = max(p[v], key=p[v].get)
            out.append((si, p[v][top], top == gold, p[v], gold))
    return out


def rank_auroc(scores, labels) -> float:
    """AUROC via the rank-sum formula, ties given their average rank."""
    pairs = sorted(zip(scores, labels))
    n, npos = len(pairs), sum(l for _, l in pairs)
    if npos in (0, n):
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        positives_in_tie = sum(1 for t in range(i, j + 1) if pairs[t][1])
        rank_sum += positives_in_tie * ((i + j) / 2 + 1)
        i = j + 1
    return (rank_sum - npos * (npos + 1) / 2) / (npos * (n - npos))


def metrics(xs) -> dict:
    n = len(xs)
    conf = [c for _, c, _, _, _ in xs]
    ok = [o for _, _, o, _, _ in xs]
    bins = [[] for _ in range(10)]
    for c, o in zip(conf, ok):
        bins[min(int(c * 10), 9)].append((c, o))
    ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins if b)
    brier = sum(sum((q - (o == g)) ** 2 for o, q in p.items()) + (0 if g in p else 1) for _, _, _, p, g in xs) / n
    nll = sum(-math.log(max(p.get(g, 0), 1e-3)) for _, _, _, p, g in xs) / n
    auroc = rank_auroc(conf, ok)
    ranked = sorted(zip(conf, ok), key=lambda t: -t[0])
    kept, right, keep95 = 0, 0, 0.0
    for c, o in ranked:
        kept += 1
        right += o
        if right / kept >= 0.95:
            keep95 = kept / n
    return {"acc": sum(ok) / n, "ECE": ece, "over": sum(conf) / n - sum(ok) / n, "Brier": brier, "NLL": nll,
            "AUROC": auroc, "keep@95": keep95, "mass_ok": sum(c for c, o in zip(conf, ok) if o) / sum(conf)}


def bootstrap(per_method: dict[str, list], n_stories: int, keys=("ECE", "AUROC"), reps=1000, seed=0) -> dict:
    """90% intervals for each method's difference from one pass, resampling stories."""
    grouped = {name: {} for name in per_method}
    for name, xs in per_method.items():
        for x in xs:
            grouped[name].setdefault(x[0], []).append(x)
    rng = random.Random(seed)
    diffs = {name: {k: [] for k in keys} for name in per_method if name != "one_pass"}
    for _ in range(reps):
        pick = [rng.randrange(n_stories) for _ in range(n_stories)]
        base = metrics([x for s in pick for x in grouped["one_pass"][s]])
        for name in diffs:
            m = metrics([x for s in pick for x in grouped[name][s]])
            for k in keys:
                diffs[name][k].append(m[k] - base[k])
    out = {}
    for name, d in diffs.items():
        out[name] = {}
        for k, v in d.items():
            v.sort()
            out[name][k] = (v[int(0.05 * reps)], v[int(0.95 * reps)])
    return out


def reliability(xs) -> list[str]:
    rows = []
    for lo in (0.0, 0.3, 0.5, 0.7, 0.9):
        hi = {0.0: 0.3, 0.3: 0.5, 0.5: 0.7, 0.7: 0.9, 0.9: 1.01}[lo]
        b = [(c, o) for _, c, o, _, _ in xs if lo <= c < hi]
        rows.append(f"{len(b):4d} @ {sum(c for c, _ in b) / len(b):.2f}->{sum(o for _, o in b) / len(b):.2f}" if b else f"{0:4d} @   -  ")
    return rows


def main():
    for k in (4, 6):
        stories, recs = load(k)
        per = {(m if est == "p" else f"{m}[{est}]"): items(stories, recs, m, est) for m, est in METHODS}
        ci = bootstrap(per, len(stories))
        print(f"\nk={k} ({len(per['one_pass'])} labels, {len(stories)} stories)")
        print(f"  {'method':18s} {'acc':>6s} {'ECE':>6s} {'over':>6s} {'Brier':>6s} {'NLL':>6s} {'AUROC':>6s} "
              f"{'keep@95':>8s} {'mass_ok':>8s}   ECE - one pass, 90% CI   AUROC - one pass, 90% CI")
        for name, xs in per.items():
            r = metrics(xs)
            line = (f"  {name:18s} {r['acc']:6.3f} {r['ECE']:6.3f} {r['over']:+6.3f} {r['Brier']:6.3f} {r['NLL']:6.3f} "
                    f"{r['AUROC']:6.3f} {r['keep@95']:8.3f} {r['mass_ok']:8.3f}")
            if name in ci:
                e, a = ci[name]["ECE"], ci[name]["AUROC"]
                line += f"   {e[0]:+.3f} to {e[1]:+.3f}         {a[0]:+.3f} to {a[1]:+.3f}"
            print(line)
        print("  reliability (labels @ mean confidence -> accuracy), bins <0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | >=0.9:")
        for name in ("one_pass", "refine1", "gibbs_T0.1[rb]", "gibbs_T0.1[vote]", "gibbs_T1[rb]"):
            print(f"    {name:18s} " + " | ".join(reliability(per[name])))


# ----------------------------------------------------------------------------- fixes for overconfidence

def relabel(label_items, conf_source):
    """Keep one method's labels, take the confidence from another method's probability of that label."""
    out = []
    for (si, _, ok, p, gold), (_, _, _, q, _) in zip(label_items, conf_source):
        top = max(p, key=p.get)
        out.append((si, q.get(top, 0.0), ok, {top: q.get(top, 0.0)}, gold))
    return out


def scaled(xs, t):
    """Rescale top-label confidence on the logit scale: c' = sigmoid(logit(c) / t)."""
    out = []
    for si, c, ok, p, gold in xs:
        c = min(max(c, 1e-4), 1 - 1e-4)
        z = math.log(c / (1 - c)) / t
        c2 = 1 / (1 + math.exp(-z))
        out.append((si, c2, ok, {max(p, key=p.get): c2}, gold))
    return out


def fit_scale(xs):
    """Pick t minimising the log loss of the correctness indicator under the rescaled confidence."""
    def loss(t):
        return -sum(math.log(max(c if ok else 1 - c, 1e-6)) for _, c, ok, _, _ in scaled(xs, t)) / len(xs)
    return min((loss(t), t) for t in [x / 20 for x in range(4, 101)])[1]


def fixes():
    print("\nFixes, scored on calibration of the label's correctness (ECE, AUROC, keep@95, mass_ok):")
    for k in (4, 6):
        stories, recs = load(k)
        one = items(stories, recs, "one_pass", "p")
        t1 = items(stories, recs, "gibbs_T1", "rb")
        for lab_m, lab_e in (("gibbs_T0.1", "vote"), ("gibbs_T0.1", "rb"), ("gibbs_T0.3", "rb")):
            lab = items(stories, recs, lab_m, lab_e)
            name = f"{lab_m}[{lab_e}]"
            rows = {f"{name}, own confidence": lab,
                    f"{name}, confidence = one pass P(label)": relabel(lab, one),
                    f"{name}, confidence = Gibbs T=1 P(label)": relabel(lab, t1)}
            # 2-fold rescaling: fit on even-numbered stories, test on odd, and the reverse
            even = [x for x in lab if x[0] % 2 == 0]
            odd = [x for x in lab if x[0] % 2 == 1]
            rows[f"{name}, own confidence rescaled (2-fold)"] = scaled(odd, fit_scale(even)) + scaled(even, fit_scale(odd))
            print(f"  k={k}")
            for rname, xs in rows.items():
                r = metrics(xs)
                print(f"    {rname:58s} acc {r['acc']:.3f}  ECE {r['ECE']:.3f}  AUROC {r['AUROC']:.3f}  "
                      f"keep@95 {r['keep@95']:.3f}  mass_ok {r['mass_ok']:.3f}")


if __name__ == "__main__":
    import sys
    fixes() if "--fixes" in sys.argv else main()
