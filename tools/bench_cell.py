#!/usr/bin/env python3
"""One comparable decode cell: serve + N streaming requests, median tok/s.

Loopback only. Starts the (signed) engine, waits /health, drives N greedy
64-token streaming requests, reports median tok/s (usage.completion_tokens /
wall) and TTFT, then stops the engine. Used to A/B thread/pinning and host-dot
variants against Colibri under the same protocol.
"""
import json, os, statistics, subprocess, sys, time, urllib.request

EXE = sys.argv[1] if len(sys.argv) > 1 else r"D:\Fennec\target\release\fennec.exe"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "D:/MEGAsync-sync/colibri-models/qwen36-35b"
BACKEND = sys.argv[3] if len(sys.argv) > 3 else "cuda"
THREADS = sys.argv[4] if len(sys.argv) > 4 else "8"
N = int(sys.argv[5]) if len(sys.argv) > 5 else 3
PORT = 8199
# The server signs every route, `/health` included, so the readiness probe has to
# carry the same key the completion request does.
AUTH = {"Authorization": "Bearer local"}

env = dict(os.environ)
proc = subprocess.Popen([EXE, "serve", "--model", MODEL, "--auto-tier",
                         "--backend", BACKEND, "--threads", THREADS,
                         "--host", "127.0.0.1", "--port", str(PORT)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
base = f"http://127.0.0.1:{PORT}"
try:
    for _ in range(60):
        try:
            urllib.request.urlopen(urllib.request.Request(base + "/health", headers=AUTH), timeout=2); break
        except Exception: time.sleep(1)
    else:
        print("engine did not become healthy"); sys.exit(2)
    toks_s, ttfts = [], []
    for i in range(N):
        body = {"model": "m", "messages": [{"role": "user", "content": "Count from 1 to 30."}],
                "max_tokens": 64, "stream": True, "temperature": 0}
        t0 = time.time(); first = None; toks = None
        r = urllib.request.urlopen(urllib.request.Request(
            base + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={**AUTH, "Content-Type": "application/json"}), timeout=180)
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            try: j = json.loads(d)
            except Exception: continue
            if first is None and j.get("choices", [{}])[0].get("delta", {}).get("content"):
                first = time.time() - t0
            if j.get("usage"): toks = j["usage"].get("completion_tokens")
        wall = time.time() - t0
        if toks: toks_s.append(toks / wall); ttfts.append(first)
    print(f"backend={BACKEND} threads={THREADS} n={len(toks_s)} "
          f"median_tok_s={statistics.median(toks_s):.3f} runs={[round(x,2) for x in toks_s]} "
          f"median_TTFT={statistics.median(ttfts):.3f}s")
finally:
    proc.terminate()
    try: proc.wait(timeout=10)
    except Exception: proc.kill()