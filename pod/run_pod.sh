#!/usr/bin/env bash
# Runs on a RunPod GPU pod (CUDA image, /workspace on a persistent volume). Writes everything to
# /workspace/out and touches /workspace/out/DONE at the end.
#   MODEL_SOURCE  weights to load (default meta-llama/Llama-3.1-8B, gated: needs HF_TOKEN). With an ungated copy
#                 such as unsloth/Meta-Llama-3.1-8B no token is needed. Either way the server presents the model
#                 as meta-llama/Llama-3.1-8B, the name the paper's code expects.
set -euo pipefail
export HF_HOME=/root/hf  # local disk: /workspace is a FUSE network volume, far too slow for imports
MODEL=meta-llama/Llama-3.1-8B
MODEL_SOURCE=${MODEL_SOURCE:-$MODEL}
COMMIT=9e964f326b1ea8f9c13d5eef70f06ac0c14cb6d3
OUT=/workspace/out
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/run_pod.log") 2>&1
rm -f "$OUT/FAILED"
trap 'rc=$?; if [ $rc -ne 0 ]; then echo "== failed (exit $rc)"; touch "$OUT/FAILED"; fi' EXIT
echo "== $(date -u) start on $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Safety net: stop this pod after 6 hours even if nobody is watching (the volume, and so the results, survive).
if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
  (sleep 21600; echo "== 6h cap reached, stopping pod"; runpodctl stop pod "$RUNPOD_POD_ID") &
else
  echo "== runpodctl or RUNPOD_POD_ID missing; no on-pod time cap"
fi

echo "== weights: $MODEL_SOURCE (served as $MODEL)"
case "$MODEL_SOURCE" in meta-llama/*) [ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN needed for $MODEL_SOURCE"; exit 1; } ;; esac
echo "== $(date -u) installing uv"; pip install -q uv

# The paper's repo at a fixed commit, with its own environment (Python 3.13).
echo "== $(date -u) repo + paper environment"
if [ ! -d /root/llg ]; then git clone -q https://github.com/hyeok9855/large-language-gibbs /root/llg; fi
cd /root/llg && git checkout -q "$COMMIT"
uv sync -q --extra consistent-reasoning
cp /workspace/pod/controls.py /root/llg/controls.py

# vLLM in its own environment, served with the paper's settings (appendix C.2).
echo "== $(date -u) vLLM environment"
if [ ! -x /root/vllm/bin/vllm ]; then uv venv -q /root/vllm --python 3.12 && VIRTUAL_ENV=/root/vllm uv pip install -q vllm; fi
# FlashInfer compiles kernels at startup and needs ninja on the PATH
[ -x /root/vllm/bin/ninja ] || VIRTUAL_ENV=/root/vllm uv pip install -q ninja
export PATH=/root/vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0  # the FlashInfer sampler JIT-compiles kernels at startup; use vLLM's own
echo "== $(date -u) starting vLLM"
/root/vllm/bin/vllm --version || true
nohup /root/vllm/bin/vllm serve "$MODEL_SOURCE" --served-model-name "$MODEL" --port 8000 \
  --enable-prefix-caching --max-model-len 16384 \
  --gpu-memory-utilization 0.85 --max-num-seqs 256 --max-num-batched-tokens 131072 > "$OUT/logs/vllm.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 180); do  # up to 30 min, but give up as soon as the server process dies
  curl -sf localhost:8000/health >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited during startup"; tail -30 "$OUT/logs/vllm.log"; exit 1; }
  sleep 10
done
curl -sf localhost:8000/health >/dev/null || { echo "vLLM did not come up"; tail -50 "$OUT/logs/vllm.log"; exit 1; }
echo "== $(date -u) vLLM up"

# Constrained decoding must be honoured, or the paper's client silently degrades: offer two answers the model
# would never give here and check the reply is one of them (or a prefix, if cut off by max_tokens).
SMOKE=$(curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",
  \"prompt\":\"Question: What colour is the sky on a clear day?\\nClaim: Blue.\\nI think this claim is\",
  \"max_tokens\":2,\"temperature\":0,\"structured_outputs\":{\"choice\":[\" cat\",\" dog\"]}}")
echo "smoke: $SMOKE" | head -c 400; echo
python3 -c "import json,sys; t=json.loads(sys.argv[1])['choices'][0]['text']; assert t and any(c.startswith(t) for c in (' cat',' dog')), t" "$SMOKE" \
  || { echo "structured_outputs not honoured by this vLLM version"; exit 1; }
echo "== $(date -u) constrained decoding ok"

run() { echo "== $(date -u) $*"; uv run python "$@"; }
for TB in truthfulQA gsm8k; do  # 1. zero-shot with the paper's own runner (expect 0.588 / 0.533)
  run consistent_reasoning/run_eval.py --testbed $TB --algorithm zeroshot --model $MODEL --port 8000 \
    --temperature 0.0 --n_partitions 1 --num_workers 128
done
# Gate: the weights must reproduce the paper's zero-shot accuracy (within 2 points), or stop here.
uv run python - <<'PY'
import glob, json, sys
paper = {"truthfulQA": 0.588, "gsm8k": 0.533}
off = []
for tb, ref in paper.items():
    acc = json.load(open(glob.glob(f"consistent_reasoning/results/{tb}/zeroshot/*/summary.json")[0]))["headline"]["mean_accuracy"]
    print(f"== zero-shot {tb}: {acc:.3f} (paper {ref:.3f})", flush=True)
    if abs(acc - ref) > 0.02:
        off.append(tb)
if off:
    print(f"== zero-shot does not match the paper for {off}; stopping", flush=True)
    sys.exit(1)
PY

mkdir -p "$OUT/paper_runs" && cp -r /root/llg/consistent_reasoning/results/. "$OUT/paper_runs/"

for TB in truthfulQA gsm8k; do  # 2. the four new arms, 5 partitions
  run controls.py --testbed $TB --model $MODEL --out "$OUT/controls"
done
for TB in truthfulQA gsm8k; do  # 3. the paper's Gibbs (m=16, hierarchical sweep), partition 0, as a second check
  run consistent_reasoning/run_eval.py --testbed $TB --algorithm gibbs --model $MODEL --port 8000 \
    --temperature 1.0 --chunk_size_cis 16 --fast_demo_order --n_partitions 5 --only_partition 0 \
    --partition_base_seed 42 --num_workers 16 || echo "== gibbs check on $TB failed (controls results kept)"
  cp -r /root/llg/consistent_reasoning/results/. "$OUT/paper_runs/"
done
echo "== $(date -u) done"
touch "$OUT/DONE"
