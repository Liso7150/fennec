#!/usr/bin/env python3
"""Client-only streaming bench: measure median tok/s against a running serve."""
import json, statistics, sys, time, urllib.request
base = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
toks = []; ttfts = []
for _ in range(n):
    body = {"model": "m", "messages": [{"role": "user", "content": "Write the numbers from 1 to 50 separated by commas, nothing else."}],
            "max_tokens": 64, "stream": True, "temperature": 0}
    t0 = time.time(); first = None; tk = None
    r = urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer local"}), timeout=180)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:"): continue
        d = line[5:].strip()
        if d == "[DONE]": break
        try: j = json.loads(d)
        except Exception: continue
        if first is None and j.get("choices", [{}])[0].get("delta", {}).get("content"): first = time.time() - t0
        if j.get("usage"): tk = j["usage"].get("completion_tokens")
    w = time.time() - t0
    if tk: toks.append(tk / w); ttfts.append(first)
print(json.dumps({"median_tok_s": round(statistics.median(toks), 3), "runs": [round(x, 2) for x in toks],
                  "median_ttft_s": round(statistics.median(ttfts), 3)}))