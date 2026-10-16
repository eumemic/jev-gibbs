#!/usr/bin/env python3
"""Context-vs-consistency controls for Large Language Gibbs (Choi et al. 2026, §5.1), on the paper's own setup.

Runs inside a clone of github.com/hyeok9855/large-language-gibbs, against a vLLM server, using the repo's
data loading, partitions (5 partitions, base seed 42, 16 questions per chunk) and prompt format. For every
claim, the other 63 claims of its chunk are shown as demonstrations (questions shuffled, claims shuffled
within a question, the target's own question last, as in the paper's Gibbs conditionals; the order of the
other questions is drawn once per target question, as in the paper's hierarchical sweep), and the model
picks " True" or " False" greedily, as in the paper's zero-shot baseline. Arms differ only in the
demonstrations' verdicts:

  zeroshot   no demonstrations (reproduces the paper's zero-shot row)
  nolabel    demonstrations with no verdict lines
  random     each demonstration's verdict a fair coin flip
  pseudo     verdicts from this script's own zero-shot answers (one refinement step, no iteration)
  gold       correct verdicts (oracle)

Compare with the paper's AR-Random and Gibbs rows for the same model (Table 1).

Usage (repo root, vLLM on :8000):
  python pod/controls.py --testbed truthfulQA --model meta-llama/Llama-3.1-8B --out results_controls
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from consistent_reasoning.dataloaders import (  # noqa: E402
    build_cid_index, build_partitions, load_eval_set, load_processed_train, select_items_for_chunk)
from consistent_reasoning.models import OpenAICompatLLM  # noqa: E402
from consistent_reasoning.prompt_utils import get_judge_prompt_fewshot  # noqa: E402

ARMS = ("nolabel", "random", "pseudo", "gold")
CHOICES = [" True", " False"]
SUFFIX = "\nI think this claim is"


def question_blocks(chunk: list[dict], rng: random.Random):
    """For each question in the chunk: the other questions' claims in one shuffled order (questions shuffled,
    claims shuffled within each), then, per target claim, its own question's other claims shuffled and last
    (paper, A.2.1). The long prefix is shared by a question's four targets, so vLLM's prefix cache reuses it,
    like the paper's hierarchical sweep."""
    by_q: dict = {}
    for it in chunk:
        by_q.setdefault(it["consistency_id"], []).append(it)
    for cid, members in by_q.items():
        others = [c for c in by_q if c != cid]
        rng.shuffle(others)
        prefix = []
        for o in others:
            m = list(by_q[o])
            rng.shuffle(m)
            prefix += m
        for target in members:
            sibs = [x for x in members if x is not target]
            rng.shuffle(sibs)
            yield target, prefix + sibs


def prompt(target: dict, demos: list[dict], verdicts: dict[int, int] | None) -> str:
    if verdicts is None:  # the demonstrations' text without their verdict lines
        body = "".join(d["prompt"][: -len(SUFFIX)] + "\n\n" if d["prompt"].endswith(SUFFIX) else d["prompt"] + "\n\n"
                       for d in demos)
        p = body + target["prompt"]
    else:
        p = get_judge_prompt_fewshot(target, [dict(d, label=verdicts[d["uid"]]) for d in demos], pipeline=False)
    return p[:-1] if p.endswith(" ") else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", choices=["truthfulQA", "gsm8k"], required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n_partitions", type=int, default=5)
    ap.add_argument("--partition_base_seed", type=int, default=42)
    ap.add_argument("--chunk_size_cis", type=int, default=16)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("results_controls"))
    args = ap.parse_args()

    cr = REPO / "consistent_reasoning"
    eval_set = load_eval_set(cr / "eval_sets" / f"{args.testbed}.json")
    ns = SimpleNamespace(testbed=args.testbed, instruction_tuned=False)
    train = load_processed_train(ns)
    cid_index = build_cid_index(train)
    partitions = build_partitions(eval_set["consistency_ids"], n_partitions=args.n_partitions,
                                  base_seed=args.partition_base_seed, chunk_size_cis=args.chunk_size_cis)
    llm = OpenAICompatLLM(model_name=args.model, base_url=f"http://localhost:{args.port}/v1", temperature=0.0,
                          max_tokens=2, instruction_tuned=False)

    def ask(p: str) -> int:
        return int(llm.generate(p, schema=CHOICES).strip() == "True")

    # Zero-shot once per claim (it does not depend on the partition); key by the source item's identity.
    items0 = [it for cid in eval_set["consistency_ids"] for it in
              select_items_for_chunk(train, cid_index, [cid], expected_group_size=eval_set["group_size"])]
    key = lambda it: it["_source_index"]  # noqa: E731  (row in the source data; set by select_items_for_chunk)
    with ThreadPoolExecutor(args.workers) as ex:
        zs = dict(zip([key(it) for it in items0], ex.map(lambda it: ask(prompt(it, [], {})), items0)))
    gold = {key(it): int(bool(it["label"])) for it in items0}
    acc = lambda pred: sum(pred[k] == gold[k] for k in pred) / len(pred)  # noqa: E731
    print(f"[{args.testbed}] zero-shot accuracy {acc(zs):.3f} on {len(zs)} claims (paper: 0.588 TQA / 0.533 GSM8K)",
          flush=True)

    results = {"testbed": args.testbed, "model": args.model, "zeroshot": acc(zs), "partitions": []}
    for p_info in partitions:
        rng = random.Random(p_info["partition_seed"])
        jobs = []
        for chunk_cids in p_info["chunks"]:
            chunk = select_items_for_chunk(train, cid_index, chunk_cids, expected_group_size=eval_set["group_size"])
            for uid, it in enumerate(chunk):
                it["uid"] = uid
            coin = {it["uid"]: rng.randint(0, 1) for it in chunk}
            verdicts = {"nolabel": None, "random": coin,
                        "pseudo": {it["uid"]: zs[key(it)] for it in chunk},
                        "gold": {it["uid"]: int(bool(it["label"])) for it in chunk}}
            for target, demos in question_blocks(chunk, rng):  # same order for every arm
                for arm in ARMS:
                    jobs.append((arm, key(target), prompt(target, demos, verdicts[arm])))
        with ThreadPoolExecutor(args.workers) as ex:
            preds = list(ex.map(lambda j: ask(j[2]), jobs))
        row = {"partition_seed": p_info["partition_seed"]}
        for arm in ARMS:
            got = {k: pr for (a, k, _), pr in zip(jobs, preds) if a == arm}
            row[arm] = acc(got)
        results["partitions"].append(row)
        print(f"[{args.testbed}] partition {p_info['partition_seed']}: "
              + "  ".join(f"{a} {row[a]:.3f}" for a in ARMS), flush=True)

    for arm in ARMS:
        xs = [r[arm] for r in results["partitions"]]
        results[f"{arm}_mean"] = statistics.mean(xs)
        results[f"{arm}_std"] = statistics.stdev(xs) if len(xs) > 1 else 0.0
    print(f"[{args.testbed}] mean over {len(partitions)} partitions: "
          + "  ".join(f"{a} {results[f'{a}_mean']:.3f}±{results[f'{a}_std']:.3f}" for a in ARMS), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"controls_{args.testbed}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
