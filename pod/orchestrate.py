#!/usr/bin/env python3
"""Run pod/run_pod.sh on a RunPod GPU pod: create the pod, wait for SSH, upload pod/, start the run, check on
it every 2 minutes, copy /workspace/out back to results/pod_out, and delete the pod.

  python3 pod/orchestrate.py --model-source unsloth/Meta-Llama-3.1-8B

On failure the logs are copied back and the pod is stopped (its volume kept for debugging). After --cap-hours
from creation the partial results are copied back and the pod is deleted. The pod id is kept in
results/pod_state.json so a rerun resumes watching the same pod. Uses .secrets/runpod_api_key and a dedicated
SSH key in .secrets/pod_key.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ROOT / ".secrets"
STATE = ROOT / "results" / "pod_state.json"
OUT = ROOT / "results" / "pod_out"
API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def api(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(API + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {(SECRETS / 'runpod_api_key').read_text().strip()}",
                                          "Content-Type": "application/json",
                                          # RunPod's Cloudflare front end rejects Python's default user agent (1010)
                                          "User-Agent": "llg-orchestrator/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"RunPod {method} {path}: HTTP {e.code} {e.read().decode(errors='replace')[:400]}") from None


class Pod:
    def __init__(self, pod_id: str):
        self.id, self.host, self.port = pod_id, None, None

    def refresh(self) -> dict:
        info = api("GET", f"/pods/{self.id}")
        ports = info.get("portMappings") or {}
        self.host, self.port = info.get("publicIp"), ports.get("22") or ports.get(22)
        return info

    def _ssh_base(self, tool: str) -> list[str]:
        opts = ["-i", str(SECRETS / "pod_key"), "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={SECRETS / 'known_hosts'}", "-o", "ConnectTimeout=20",
                "-o", "ServerAliveInterval=30", "-o", "BatchMode=yes"]
        return [tool, *opts, "-P" if tool == "scp" else "-p", str(self.port)]

    def ssh(self, cmd: str, check: bool = False) -> subprocess.CompletedProcess:
        # -n: never hold stdin, or ssh waits on background processes started by cmd
        return subprocess.run(self._ssh_base("ssh") + ["-n", f"root@{self.host}", cmd], capture_output=True,
                              text=True, timeout=300, check=check)

    def fetch(self, remote: str, local: Path) -> bool:
        local.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(self._ssh_base("scp") + ["-r", f"root@{self.host}:{remote}", str(local)],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode:
            log(f"scp failed: {r.stderr.strip()[:300]}")
        return r.returncode == 0


def create(args) -> Pod:
    body = {"name": "llg-controls", "imageName": IMAGE, "gpuTypeIds": [args.gpu], "gpuCount": 1,
            "cloudType": args.cloud, "interruptible": False, "containerDiskInGb": 100, "volumeInGb": 20,  # software and weights on local disk
            "volumeMountPath": "/workspace", "ports": ["22/tcp"], "supportPublicIp": True,
            "env": {"PUBLIC_KEY": (SECRETS / "pod_key.pub").read_text().strip(), "MODEL_SOURCE": args.model_source}}
    if args.model_source.startswith("meta-llama/"):
        body["env"]["HF_TOKEN"] = (SECRETS / "hf_token").read_text().strip()
    t_end = time.time() + args.wait_for_gpu_min * 60
    while True:  # capacity comes and goes; retry the same request rather than changing it
        try:
            info = api("POST", "/pods", body)
            break
        except RuntimeError as e:
            if "no instances currently available" not in str(e) or time.time() > t_end:
                raise
            log("no machine available yet; retrying in 2 minutes")
            time.sleep(120)
    pod = Pod(info["id"])
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({"id": pod.id, "created": time.time(), "gpu": args.gpu,
                                 "cost_per_hr": info.get("costPerHr"), "model_source": args.model_source}))
    log(f"created pod {pod.id} ({args.gpu}, {args.cloud.lower()} cloud, ${info.get('costPerHr')}/h)")
    return pod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-source", default="unsloth/Meta-Llama-3.1-8B")
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3")
    ap.add_argument("--cloud", choices=["COMMUNITY", "SECURE"], default="COMMUNITY")
    ap.add_argument("--cap-hours", type=float, default=6.0)
    ap.add_argument("--wait-for-gpu-min", type=float, default=30, help="keep retrying creation this long")
    args = ap.parse_args()

    state = json.loads(STATE.read_text()) if STATE.exists() else None
    if state:
        pod = Pod(state["id"])
        log(f"resuming pod {pod.id}")
    else:
        pod = create(args)
        state = json.loads(STATE.read_text())
    deadline = state["created"] + args.cap_hours * 3600

    if pod.refresh().get("desiredStatus") == "EXITED":  # stopped after a failure: start it again (volume kept)
        api("POST", f"/pods/{pod.id}/start")
        log("restarted stopped pod")
    for _ in range(90):  # up to 30 minutes for the pod to come up with SSH
        info = pod.refresh()
        if pod.host and pod.port and pod.ssh("true").returncode == 0:
            break
        log(f"waiting: status={info.get('desiredStatus')} ip={pod.host} ssh_port={pod.port}")
        time.sleep(20)
    else:
        log("pod never became reachable over SSH; deleting it")
        api("DELETE", f"/pods/{pod.id}")
        sys.exit(1)
    log(f"ssh ok: root@{pod.host} -p {pod.port}")

    started = pod.ssh("test -f /workspace/out/logs/run_pod.log && ! test -f /workspace/out/FAILED").returncode == 0
    if not started:  # never started, or the last attempt failed: (re)start with the current pod/ files
        r = subprocess.run(pod._ssh_base("scp") + ["-r", str(ROOT / "pod"), f"root@{pod.host}:/workspace/"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise RuntimeError(f"upload failed: {r.stderr[:300]}")
        # ';' not '&&': with 'cd && cmd &' the whole list runs in a subshell that keeps ssh's channel open
        pod.ssh(f"cd /workspace; MODEL_SOURCE={args.model_source} setsid nohup bash /workspace/pod/run_pod.sh "
                "> /workspace/run_pod.nohup 2>&1 < /dev/null &", check=True)
        log("run started")

    seen = 0
    while True:
        time.sleep(120)
        r = pod.ssh("cat /workspace/out/logs/run_pod.log 2>/dev/null | grep -E '^== |partition|accuracy|mean over' ; "
                    "test -f /workspace/out/DONE && echo __DONE__; test -f /workspace/out/FAILED && echo __FAILED__; "
                    "true")  # exit 0 whenever ssh itself worked
        if r.returncode != 0:
            log(f"ssh check failed ({r.stderr.strip()[:120]}); retrying")
            if time.time() > deadline:
                log("cap reached while unreachable; deleting pod")
                api("DELETE", f"/pods/{pod.id}")
                sys.exit(1)
            continue
        lines = [l for l in r.stdout.splitlines() if not l.startswith("__")]
        for l in lines[seen:]:
            log(f"pod: {l}")
        seen = len(lines)
        if "__DONE__" in r.stdout:
            log("run finished; copying results")
            if pod.fetch("/workspace/out", OUT):
                api("DELETE", f"/pods/{pod.id}")
                STATE.unlink(missing_ok=True)
                log(f"results in {OUT.relative_to(ROOT)}; pod deleted")
                return
            log("could not copy results; stopping the pod instead of deleting it (volume kept)")
            api("POST", f"/pods/{pod.id}/stop")
            sys.exit(1)
        if "__FAILED__" in r.stdout:
            log("run_pod.sh stopped without finishing; copying logs and stopping the pod (volume kept)")
            pod.fetch("/workspace/out", OUT)
            api("POST", f"/pods/{pod.id}/stop")
            sys.exit(1)
        if time.time() > deadline:
            log(f"{args.cap_hours} h cap reached; copying partial results and deleting the pod")
            pod.fetch("/workspace/out", OUT)
            api("DELETE", f"/pods/{pod.id}")
            STATE.unlink(missing_ok=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
