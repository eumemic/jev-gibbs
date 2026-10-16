# Jev as the conditional in Large Language Gibbs (TruthfulQA, §5.1)

Replication of the consistent-reasoning experiment in Choi, Gouk & Whitammer, *Structured Inference with
Large Language Gibbs* (arXiv 2606.19264), with TypeSafe's Jev (`jev-latest` = jev-1.13.0) supplying the
per-claim conditionals. The dataset is the paper's eval set (256 questions × 4 claims), and seed `p` uses
the same partition as the paper's partition `p` (`random.Random(42 + p)`).

## Files

| File | What it is |
|---|---|
| `jev_gibbs.py` | TruthfulQA and GSM8K with Jev (`--task`). Modes: `zeroshot`, `probe`, `context` (no-verdict control), `gibbs` (`--schedule block\|seq`) |
| `analyze.py` | Summary table and per-seed paired comparisons for the TruthfulQA runs |
| `switches.py` | How often, and where, Jev changed its answer inside a logged chain |
| `mask_probe.py` | Whether a question can mask or override part of a shared state |
| `kk.py` | Knights-and-knaves puzzles: screen and exact Gibbs |
| `er.py` | Entity resolution on MusicBrainz 20K: screen and clustering baselines |
| `kin.py` | Kinship chains: screen, Gibbs, and the paired `compare` mode |
| `calib.py` | Calibration of the kinship labels' confidences (`--fixes` for independent and rescaled confidences) |
| `pod/` | The deciding experiment on the paper's model: `run_pod.sh` (runs on the GPU pod), `controls.py` (the four arms), `orchestrate.py` (creates, watches and deletes the RunPod pod) |
| `truthfulqa_eval.json`, `gsm8k_eval.json` | The paper's eval sets (256 questions × 4 claims), built from its repo (MIT) |
| `jev_gibbs.draft.py` | The earlier draft, kept for reference |
| `results/`, `logs/` | Raw outputs of every run below; `results/pod_out/` holds the RunPod run |

Not in the repo:
- `.secrets/`: API keys and the pod's SSH key.
- `data/musicbrainz/`: fetch it with
  `mkdir -p data/musicbrainz && curl -L -o data/musicbrainz/musicbrainz-20-A01.csv.dapo https://dbs.uni-leipzig.de/files/datasets/saeedi/musicbrainz-20-A01.csv.dapo`.
- The paper runner's zero-shot chunk caches (their `summary.json` files are kept) and the vLLM server log.

```bash
export TYPESAFE_API_KEY=...
python3 jev_gibbs.py zeroshot
python3 jev_gibbs.py context --m 16 --shuffles 5
python3 jev_gibbs.py gibbs --m 16 --schedule block --seeds 5
python3 analyze.py
```

## Results

Accuracy is mean ± sd over the paper's 5 partitions; the other metrics are averaged over them. Calls and
cost are per seed (one labelling of all 1,024 claims). "rb" averages Jev's conditional probabilities after
burn-in; "vote" is the paper's majority vote over 25 sampled labels.

| Method | Accuracy | Brier | ECE | AUROC | Calls | Cost |
|---|---|---|---|---|---|---|
| Jev zero-shot | 0.913 | 0.074 | 0.060 | 0.955 | 16 | $0.005 |
| Context, no verdicts, m=16, 1 pass | 0.931 ± 0.001 | 0.058 | 0.052 | 0.970 | 64 | $0.01 |
| Context, no verdicts, m=4, 5 passes | 0.924 ± 0.003 | 0.060 | 0.055 | 0.967 | 1,280 | $0.06 |
| Gibbs m=1 block, rb | 0.919 ± 0.001 | 0.064 | 0.051 | 0.961 | 35,840 | $0.82 |
| Gibbs m=4 block, rb | 0.928 ± 0.003 | 0.060 | 0.056 | 0.968 | 8,960 | $0.48 |
| Gibbs m=16 block, rb | 0.931 ± 0.003 | 0.057 | 0.056 | 0.970 | 2,240 | $0.40 |
| Gibbs m=16 block, vote | 0.927 ± 0.006 | 0.060 | 0.049 | 0.968 | 2,240 | $0.40 |
| Gibbs m=16 seq (exact), rb, 1 seed | 0.933 | 0.058 | 0.054 | 0.969 | 35,840 | $4.06 |
| Gibbs m=16, 4 sweeps from zero-shot, rb | 0.928 ± 0.002 | 0.059 | 0.053 | 0.967 | 272 | $0.05 |

Paper, TruthfulQA: best result 0.835 (OLMo-3.1-32B-Instruct, Gibbs m=16); base models 0.744–0.776 at m=16;
zero-shot 0.588–0.755. `python3 analyze.py` prints every run, including the paper's full rows.

## Findings

1. **Jev alone beats every number in the paper.** Zero-shot Jev scores 0.913 for $0.005 and about one
   second of wall time. The paper's best method reaches 0.835.
2. **The gain over zero-shot comes from showing Jev related claims, not from label consistency.** One pass
   in which each claim sees the rest of its chunk *without verdicts* scores 0.931 at m=16, the same as
   Gibbs, with the same calibration, using 35× fewer calls than block Gibbs and 560× fewer than exact
   sequential Gibbs. Against a control that averages 5 shuffled passes, Gibbs (rb) gains +0.1 to +0.4
   points (about 1–4 claims of 1,024), consistently across seeds. Majority vote gains nothing.
3. **Why the chain does little (corrected).** Jev does respond to labels, strongly, on the claims it is
   unsure about. In a logged m=16 chain (`switches.py`), 24% of claim-chains ever had P(true) in 0.3–0.7.
   In 10.8% (176 distinct claims) Jev's answer crossed 0.5 as the sibling labels changed, with swings of up
   to 0.82. Its P moved toward the gold answer when the siblings were mostly labelled correctly in 77% of
   cases. On contested questions ("What do white rabbits carry?", "count your chickens before they hatch")
   the chain moves between two self-consistent readings, literal and idiomatic. Because the labels are Jev's
   own, nothing breaks the tie, and averaging over the two readings lands near the single-pass answer. The
   other 76% of claims are confident and never get a real chance to switch. The earlier "0.08–0.10 average
   shift" figure was averaged mostly over those confident claims. The paper's base models start at 59–61%
   and gain mainly from their own pseudo-labelled demonstrations teaching them the task, which Jev gets from
   the instructions and criteria.
4. **Confidence.** Jev's raw probabilities are already calibrated (ECE 0.05–0.06), and the most confident
   half of claims is 97–98% correct in every run except the 3-sample vote. Gibbs does not improve calibration. If you do run a
   chain, average the probabilities rather than voting: with 3 kept sweeps, voting gives 0.901 and
   averaging 0.928.
5. **Block vs exact Gibbs.** The 4-colour block schedule (one claim per question updated per call, all of
   them removed from a shared state) is an approximation, but it matched exact sequential Gibbs on
   accuracy (both 0.933 averaged and 0.924 by vote on the same partition) at a tenth of the cost.

## Can a question mask part of the shared state? (`mask_probe.py`, 256 claims)

A request has one `state`, billed once, and every question reads all of it. The API has no per-question
masking. The only lever is the question's `instructions` (backtick paths into the state, structured
fields). Measured with an m=16 context labelled by zero-shot verdicts:

| Mechanism | Result |
|---|---|
| Target's own verdict left in the state, no mask | P moves 0.26 between showing True and False; accuracy 0.867 vs 0.938 |
| "Ignore the verdict recorded for this claim" | Leak halves to 0.12, still larger than the whole sibling signal (0.08) |
| Same, pointing at the target by state path | Leak 0.19 (indirection hurts, as the jaggedness docs warn) |
| "Treat these sibling verdicts as flipped" (override) | Right on average (1.09× a real edit) but noisy: mean error 0.09, the size of the effect |
| Each question carries its own context in `instructions` | Exact and batchable across chains, coupling kept (1.12×), but accuracy 0.906 vs 0.938 (zero-shot 0.898 on the same claims), and no token saving |

Even a perfect mask would not give sequential Gibbs inside one request: all questions in a request read the
same label snapshot, so their updates are simultaneous. The only hard mask is deleting entries from the
shared state, which leads to the colour-class (block) schedule used above.

## Changes from the draft

- **Criteria.** TruthfulQA labels non-committal answers such as "I have no comment" as true. The draft's
  criteria ("factually true, accurate answer") scored 0.873 zero-shot; criteria that state the convention
  score 0.913. Both are above the paper's best.
- **No-verdict control.** Added the control in finding 2. Without it, the Gibbs gain over zero-shot looks
  like a consistency effect.
- **Exact sequential schedule.** Added `--schedule seq`, which follows the paper: one variable per step, a
  fresh order each sweep, the target's own question last. The draft's docstring presented its block update
  as Gibbs.
- **Partitions and seeds.** The draft used its own partitioning and one seed; runs now use the paper's 5
  partitions.
- **Client.** It now retries `529 Overloaded` and honours `retry-after`, and its usage counters are
  thread-safe.
- **State format.** The state is structured JSON, following the docs. The effect of this change wasn't
  measured.

## Caveats

- TruthfulQA has been public since 2021, so Jev's accuracy may partly reflect training exposure. Check on
  your own data before relying on these numbers.
- Jev returns probabilities rounded to two decimals.
- Exact sequential Gibbs was run on one seed only ($4 per seed).
- `jev-preview` (1.14) scores 0.915 zero-shot. The masking probe was not rerun on it.
- Total API spend for everything in `results/`: $16.54.

## Looking for a task where Gibbs should help Jev

The most a chain can add over one pass is roughly: accuracy when each item is shown the other items with
their *correct* labels, minus accuracy when it is shown them without labels. If that gap is near zero,
the labels carry nothing the items' text doesn't, and no chain can beat a single pass with context.

| Task | One pass, no labels | Neighbours' correct labels given | Gap | Notes |
|---|---|---|---|---|
| TruthfulQA (§5.1) | 91.8% (m=16 context; same 256 probe claims) | 94.5% | +2.7 | Small but real; Gibbs realised +0.1–0.4 of it. Siblings only: 91.0% → 92.6% |
| GSM8K-Verification (`--task gsm8k`) | 93.8% (siblings shown) | 93.8% | 0 | A no-model rule ("the final answer that appears twice is right") scores 89.7%, above the paper's base-model Gibbs results; Jev zero-shot 87.4% |
| Knights and knaves (`kk.py`, generated, unique solutions) | 47–53% (chance) | 83–92% | +30–40 | Large gap |

Gibbs on knights and knaves (exact sequential, 50 puzzles per size):

| People | One pass | Gibbs (rb) | Puzzles solved, one pass → Gibbs |
|---|---|---|---|
| 3 | 51% | 64% | 24% → 48% |
| 4 | 50% | 62% | 8% → 38% |
| 6 | 47% | 59% | 4% → 10% |
| 8 | 53% | 53% | 2% → 2% |

The gap is real, but Gibbs only cashes it in for small puzzles. Even with correct neighbours, each step is
83–92% right, and errors cluster on steps that need two inferences: 70–72% when a knave's statement is
about a pair or a knave calls someone a knight, against 90–95% for single-inference steps. A sweep
re-judges every person, so the solution survives a sweep with probability about (step accuracy)^N, and
the chain keeps leaving it. A task that suits Jev + Gibbs needs a large gap and steps of a single
inference that Jev gets right about 97%+ of the time given correct neighbours. It should also be natural
language rather than formal logic, since code can solve knights and knaves outright.

### Entity resolution: MusicBrainz 20K (`er.py`)

Data: `musicbrainz-20-A01.csv.dapo` from the Leipzig entity-resolution benchmarks, in `data/musicbrainz/`.
It has 19,375 song records from 5 deduplicated sources, with 10,000 songs of 1–5 records each, deliberately
corrupted. Each block is one song cluster of 3–5 records plus look-alike clusters that share title, artist or
album words, 14 records at most. The variables are cross-source record pairs: 60 blocks, 4,393 pairs, 831
matches.

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Jev, the two records alone | 0.985 | 0.995 | 0.990 |
| Jev, whole block visible | 0.996 | 0.993 | 0.995 |
| Jev, its own decisions on the other pairs given (one refinement step) | 1.000 | 0.988 | 0.994 |
| Jev, correct decisions on the other pairs given (oracle) | 0.999 | 0.993 | 0.996 |
| Records alone + transitive closure | 0.962 | 1.000 | 0.981 |
| Records alone + correlation clustering (one record per source per cluster) | 0.998 | 1.000 | 0.999 |

Jev saturates this benchmark, and the oracle gap is 0.001 F1. The best result is Jev's pairwise
probabilities plus correlation clustering in code, with no chain. Cost: $1.08.

### Kinship chains (`kin.py`, generated)

Each story comes from a fresh random family tree (three generations with in-laws, about 21 people) and a
random path of k relatives. The story states only the k consecutive relations, in varied phrasings and
shuffled order. The variables are the relations between all non-adjacent pairs on the path, one Choice each
over 23 kinship terms (about 15 per story at k=6). End-to-end means the relation between the two ends of the
path. 40 stories per setting.

Screen, end-to-end accuracy:

| k | One pass | Oracle, "take as correct" | Random labels, "take as correct" | Random labels, "tentative guesses" | One step with own labels, "tentative" |
|---|---|---|---|---|---|
| 3 | 65% | 98% | 12% | – | – |
| 4 | 45% / 35%* | 100% | 8% | 25% | 52% |
| 6 | 20% / 30%* | 88% | 8% | 25% | 48% |

\* The two numbers come from different story samples (see the caveat below).

Phrasing the labels as "tentative guesses; the story is authoritative" keeps most of the oracle gain
(per-variable accuracy 84% vs 59% at k=4) while wrong labels cost 8–13 points instead of 39–46. Showing only
the labels that involve the two people being asked about was worse.

Gibbs, paired with the one pass on the same stories within each run:

| Variant | k | One pass, per variable / end-to-end | Gibbs, per variable / end-to-end |
|---|---|---|---|
| "take as correct", T=1 | 4 | 55% / 35% | 29% / 30% (collapses) |
| "tentative", T=1 | 4 | 61% / 45% | 56% / 50% |
| "tentative", T=1 | 6 | 43% / 12% | 37% / 18% |
| "tentative", T=0.3 | 4 | 68% / 62% | 67% / 65% |
| "tentative", T=0.3 | 6 | 48% / 28% | 46% / 45% |

The sharpened chain (T=0.3) gains 17 points on the hardest question (the 6-step relation) and stays level
per variable. A single refinement step with Jev's own best guesses gained about as much end-to-end in the
screen, for about 1/25 of the calls. Whether the chain beats one step needs both run on the same stories.

Caveat: until it was fixed, `make_story` picked path steps from a Python set, whose
order varies between processes, so each run drew different stories. Compare only within a run. Stories are
now reproducible. The T=0.1 run stopped when the account ran out of credits (HTTP 402).

#### Same-story comparison (`python3 kin.py compare --k 4 6 --stories 100`)

100 reproducible stories per length. Every method starts from the same one-pass answers and sees the same
option order, and labels are shown as tentative guesses. Gibbs runs 5 burn-in and 20 kept sweeps; "vote"
is the most frequent sampled label, "rb" the highest averaged probability. p-values are exact paired sign
tests. Cost: $6.01 for 159,192 calls (about $0.006 per 4-step story for Gibbs).

| k=4 | Per relation | End-to-end | Stories fully solved | Calls per story |
|---|---|---|---|---|
| One pass | 65.4% | 48% | 25% | 1 |
| One refinement step | 65.6% | 56% | 33% | 6 |
| Gibbs T=1 (rb) | 68.3% | 64% | 34% | 150 |
| Gibbs T=0.3 (rb) | 69.4% | 64% | 44% | 150 |
| Gibbs T=0.1 (vote) | 70.1% | 64% | 46% | 150 |

| k=6 | Per relation | End-to-end | Stories fully solved | Calls per story |
|---|---|---|---|---|
| One pass | 47.4% | 23% | 0% | 1 |
| One refinement step | 47.5% | 34% | 1% | 15 |
| Gibbs T=1 (rb) | 44.6% | 39% | 2% | 373 |
| Gibbs T=0.3 (vote) | 47.2% | 40% | 4% | 373 |
| Gibbs T=0.1 (vote) | 47.7% | 37% | 5% | 373 |

Paired tests for sharpened Gibbs (T=0.1 or 0.3):

- **Against one pass, end-to-end:** +16 points at k=4 (p≈0.005) and +12–17 at k=6 (p 0.002–0.015).
- **Against one refinement step, at k=4:** per relation +4–5 points (p 0.002–0.015); stories fully solved
  33% → 44–46% (+12–14 stories against −1, p≤0.003); end-to-end +8 points (p≈0.12–0.15, not significant).
- **Against one refinement step, at k=6:** no significant difference on any measure. Per relation is level,
  end-to-end +1–6 points, stories solved +3–4 (p≥0.125).

T=1 sampling is worse than sharpened sampling, and at k=6 it is significantly worse per relation than one
refinement step (p=0.007).

So this is the regime where Jev + Gibbs earns its cost. With 4-step chains, the chain makes the labels
significantly more accurate and more consistent across the whole family than one pass or one refinement
step. At 6 steps, per-step errors (73% accuracy on one-step compositions, from the screen) still limit it.

#### Are the chain's confidences usable as training weights? (`calib.py`)

Confidence is the probability of the chosen label. ECE is the mean gap between confidence and accuracy, and
AUROC measures how well confidence separates right labels from wrong ones. keep@95 is the share of labels
kept when keeping the most confident while their accuracy stays at or above 95%. mass_ok is the share of
total confidence that sits on correct labels, which is what a confidence-weighted loss trains on. Same
100-story runs; differences from one pass have 90% story-bootstrap intervals in `python3 calib.py`.

| Labels and confidence | k=4 acc | ECE | AUROC | keep@95 | mass_ok | k=6 acc | ECE | AUROC | mass_ok |
|---|---|---|---|---|---|---|---|---|---|
| One pass | 65.4% | 0.030 | 0.81 | 28% | 0.74 | 47.4% | 0.085 | 0.73 | 0.55 |
| One refinement step | 65.6% | 0.098 | 0.82 | 36% | 0.72 | 47.5% | 0.184 | 0.70 | 0.53 |
| Gibbs T=1, averaged | 68.3% | 0.044 | 0.79 | 35% | 0.76 | 44.6% | 0.125 | 0.69 | 0.51 |
| Gibbs T=0.1, own vote fraction | 70.1% | 0.267 | 0.58 | 0% | 0.71 | 47.7% | 0.457 | 0.57 | 0.49 |
| Gibbs T=0.1 labels, one-pass P(label) | 70.1% | 0.126 | 0.79 | 28% | **0.82** | 47.7% | 0.084 | 0.73 | **0.61** |
| Gibbs T=0.1 labels, Gibbs T=1 P(label) | 70.1% | 0.074 | 0.78 | **36%** | 0.79 | 47.7% | 0.118 | 0.68 | 0.56 |
| Gibbs T=0.1 vote, rescaled on half the stories | 70.1% | 0.117 | 0.58 | 0% | 0.71 | 47.7% | 0.309 | 0.57 | 0.49 |

- **Conditioning on its own labels makes Jev overconfident.** One refinement step adds about +0.07–0.10
  of overconfidence, and sharpened chains more. Their vote fractions put 82–90% of labels at ≥0.9
  confidence (72% accurate at k=4, 51% at k=6), so they can't separate right from wrong (AUROC 0.57–0.58).
- **Use the chain for the labels and an independent source for the weights.** Taking the confidence from
  one pass's probability of the chain's label gives the highest share of weight on correct labels at both
  lengths (0.82 and 0.61), and the best discrimination at k=6. Taking it from a T=1 chain is best
  calibrated at k=4 and keeps the most labels at ≥95% accuracy.
- **Rescaling on labelled stories fixes the level but not the ranking.** AUROC is unchanged, so rescaling
  can't rescue vote fractions.

## Deciding experiment on the paper's own model (`pod/`, RunPod H100)

Llama-3.1-8B base (Unsloth's ungated copy, served under the official name), vLLM 0.31 with the paper's
settings, the paper's repo at commit `9e964f3`, its prompt format and its 5 partitions (16 questions per
chunk). For each claim, the other 63 claims of its chunk are shown, ordered as in the paper's Gibbs
conditionals, and the model answers greedily. The arms differ only in the verdicts shown:

| Arm (m=16) | TruthfulQA | GSM8K-Verification |
|---|---|---|
| Zero-shot (paper's runner, our pod) | 0.591 (paper 0.588) | 0.533 (paper 0.533) |
| Context, no verdicts | 0.570 ± 0.006 | 0.663 ± 0.011 |
| Context, random verdicts | 0.644 ± 0.022 | 0.673 ± 0.027 |
| Context, the model's own zero-shot verdicts (one step) | 0.648 ± 0.013 | 0.673 ± 0.006 |
| AR-Random (paper, Table 1) | 0.720 | 0.792 |
| Gibbs (paper's runner, our pod, partition 0) | 0.755 | 0.893 |
| Gibbs (paper, Table 1 / Table 3 same sweep) | 0.744 / 0.753 | 0.888 / 0.890 |
| Context, correct verdicts (oracle) | 0.909 ± 0.011 | 0.938 ± 0.007 |

Means ± sd over the 5 partitions; the Gibbs check is one partition.

- **Context alone does not explain the paper's gains.** Without verdicts, TruthfulQA accuracy is slightly
  below zero-shot. GSM8K gains 13 points, about a third of Gibbs's 36.
- **Format matters on TruthfulQA.** Verdict lines help even when random: +7 points over no verdicts. The
  model's own zero-shot verdicts, about 59% correct, do no better than random ones. On GSM8K format adds
  about 1 point.
- **Iteration matters for this model.** Gibbs beats one step of its own verdicts by 11 points on TruthfulQA
  and 22 on GSM8K. The paper's claim that iterating labels helps holds for weak base models. On GSM8K,
  Gibbs (0.893) lands at the no-model answer-agreement rule (0.897), so its gain there may be the dataset's
  construction.
- **The oracle gap separates weak and strong judges.** Correct verdicts are worth +34 (TruthfulQA) and +28
  (GSM8K) points over no verdicts for Llama-8B, against +2.7 and 0 for Jev. A strong judge already gets
  from the claims what the labels would tell a weak one.

Cost: about 3.3 GPU-hours of a secure-cloud H100 at $3.99/hour, about $13, including two failed starts.
