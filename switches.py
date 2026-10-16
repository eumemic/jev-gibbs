#!/usr/bin/env python3
"""How often did Jev change its answer inside the Gibbs chains, and why? Reads gibbs_m16_block_logged.json."""
import json
from collections import Counter
from pathlib import Path
from jev_gibbs import load_data

HERE = Path(__file__).resolve().parent
by_q = load_data(HERE / "truthfulqa_eval.json")
claim = {c["uid"]: c for q in by_q.values() for c in q}
sibs = {c["uid"]: [s["uid"] for s in by_q[c["qid"]] if s is not c] for c in claim.values()}
r = json.loads((HERE / "results/gibbs_m16_block_logged.json").read_text())
BURN = int(r["args"]["burn_in"])

rows = []  # one per (seed, claim): its post-burn-in conditionals and the sibling labels it saw
for seed, chains in r["traces"].items():
    for trace in chains:
        uids = [int(u) for u in trace[0]["p"]]
        for u in uids:
            ps, sib_ok = [], []
            for t in range(BURN, len(trace)):
                ps.append(trace[t]["p"][str(u)])
                prev = trace[t - 1]["labels"]  # labels going into this sweep (approximation for block order)
                sib_ok.append(sum(prev[str(s)] == claim[s]["label"] for s in sibs[u]))
            rows.append({"seed": seed, "uid": u, "ps": ps, "sib_ok": sib_ok})

n = len(rows)
crossed = [x for x in rows if min(x["ps"]) < 0.5 < max(x["ps"])]
rng = [max(x["ps"]) - min(x["ps"]) for x in rows]
uncertain = [x for x in rows if any(0.3 <= p <= 0.7 for p in x["ps"])]
print(f"{n} claim-chains (1,024 claims x 5 seeds), {len(rows[0]['ps'])} post-burn-in conditionals each "
      f"= {n * len(rows[0]['ps']):,} conditionals")
print(f"  ever in 0.3-0.7 (a real chance to switch): {len(uncertain)} ({100 * len(uncertain) / n:.1f}%)")
print(f"  Jev's answer crossed 0.5 at least once:    {len(crossed)} ({100 * len(crossed) / n:.1f}%), "
      f"covering {len({x['uid'] for x in crossed})} distinct claims")
print(f"  range of P within a chain: median {sorted(rng)[n // 2]:.2f}, 90th pct {sorted(rng)[int(.9 * n)]:.2f}, "
      f"max {max(rng):.2f}")
buckets = Counter(min(int(x * 10), 9) for x in rng)
print("  histogram of range:", " ".join(f"{b / 10:.1f}+:{buckets[b]}" for b in range(10)))

# Does P move with the sibling labels, and in the right direction? Within each claim-chain, compare the
# mean P toward the gold answer when more siblings are labelled correctly vs fewer.
def toward_gold(u, p):
    return p if claim[u]["label"] else 1 - p
pairs = []
for x in crossed:
    hi = [toward_gold(x["uid"], p) for p, k in zip(x["ps"], x["sib_ok"]) if k >= 2]
    lo = [toward_gold(x["uid"], p) for p, k in zip(x["ps"], x["sib_ok"]) if k <= 1]
    if hi and lo:
        pairs.append(sum(hi) / len(hi) - sum(lo) / len(lo))
if pairs:
    pos = sum(d > 0 for d in pairs)
    print(f"\nAmong switching claim-chains seen with both mostly-right and mostly-wrong sibling labels (n={len(pairs)}):")
    print(f"  P toward the gold answer is higher when siblings are mostly right in {pos}/{len(pairs)}; "
          f"mean difference {sum(pairs) / len(pairs):+.3f}")

# Where: claims that switched in the most seeds, with their P range and the sibling claims.
by_uid = Counter(x["uid"] for x in crossed)
print("\nClaims whose answer crossed 0.5 in the most seeds:")
for u, k in by_uid.most_common(15):
    xs = [x for x in crossed if x["uid"] == u]
    lo, hi = min(min(x["ps"]) for x in xs), max(max(x["ps"]) for x in xs)
    c = claim[u]
    print(f"  {k}/5 seeds  P {lo:.2f}-{hi:.2f}  gold={'T' if c['label'] else 'F'}  {c['question'][:55]!r} -> {c['claim'][:70]!r}")
