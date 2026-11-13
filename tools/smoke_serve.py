#!/usr/bin/env python3
"""Smoke-test the Fennec HTTP surface against the contract the suite spawns.

Loopback only. Waits for /health, then exercises every route fennec-core's
EngineClient uses, and finally drives the real client itself so the check is
the contract rather than my reading of it.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8184"
KEY = "local"
FAILS: list[str] = []
PASSES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSES if ok else FAILS).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  :: {detail}" if detail and not ok else ""),
          flush=True)


def req(path: str, body=None, method=None, key=KEY, accept=None, timeout=180):
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    if accept:
        headers["Accept"] = accept
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, headers=headers,
                               method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def wait_health(limit_s: float = 900.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        try:
            code, _, raw = req("/health", timeout=3)
            if code == 200:
                print(f"healthy after {time.time() - t0:.0f}s: {raw.decode()[:200]}", flush=True)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> int:
    if not wait_health():
        print("engine never became healthy", flush=True)
        return 1

    code, _, raw = req("/health")
    h = json.loads(raw)
    check("GET /health", code == 200 and h.get("status") == "ok" and h.get("engine") == "fennec",
          f"{code} {raw[:200]}")

    code, _, _ = req("/health", key=None)
    check("GET /health without a key is 401", code == 401, str(code))
    code, _, _ = req("/health", key="wrong")
    check("GET /health with a wrong key is 401", code == 401, str(code))

    # urllib capitalises header names, so a fully lower-case request needs a
    # raw socket. This is the regression: the server used to lower-case the
    # whole header block, which folded "Bearer" and rejected every client.
    import socket
    with socket.create_connection(("127.0.0.1", 8184), timeout=10) as s:
        s.sendall(b"GET /health HTTP/1.1\r\nhost: 127.0.0.1:8184\r\n"
                  b"authorization: bearer local\r\nconnection: close\r\n\r\n")
        raw = b""
        while True:
            b = s.recv(65536)
            if not b:
                break
            raw += b
    check("lower-case header names and bearer scheme are accepted",
          raw.startswith(b"HTTP/1.1 200"), raw[:120].decode("utf-8", "replace"))

    code, _, raw = req("/v1/models")
    m = json.loads(raw)
    model = m["data"][0]["id"]
    check("GET /v1/models owned_by fennec",
          code == 200 and m["data"][0]["owned_by"] == "fennec" and bool(model),
          f"{code} {raw[:200]}")
    check("GET /v1/models carries max_model_len", "max_model_len" in m["data"][0])

    for path, keys in (("/metrics", ("requests",)), ("/experts", ("layers",)), ("/profile", ())):
        code, _, raw = req(path)
        ok = code == 200
        if ok and keys:
            try:
                ok = all(k in json.loads(raw) for k in keys)
            except Exception:
                ok = False
        check(f"GET {path}", ok, f"{code} {raw[:200]}")

    code, hdrs, _ = req("/v1/chat/completions", method="OPTIONS", key=None)
    check("OPTIONS preflight is 204 without a key", code == 204, str(code))
    check("OPTIONS allows Authorization",
          "authorization" in (hdrs.get("Access-Control-Allow-Headers") or "").lower())

    msgs = [{"role": "user", "content": "Reply with the single word: ok"}]
    # Thinking is on by default, and a thinking model spends a small budget
    # entirely inside the think block - content is then legitimately empty.
    # Ask for the answer channel explicitly when asserting on content.
    body = {"model": model, "messages": msgs, "max_tokens": 24, "temperature": 0.0,
            "enable_thinking": False}
    t0 = time.time()
    code, _, raw = req("/v1/chat/completions", body)
    dt = time.time() - t0
    c = json.loads(raw) if code == 200 else {}
    ch = (c.get("choices") or [{}])[0]
    check("POST /v1/chat/completions (JSON)",
          code == 200 and ch.get("message", {}).get("role") == "assistant"
          and isinstance(ch.get("message", {}).get("content"), str)
          and ch.get("finish_reason") in ("stop", "length"),
          f"{code} {raw[:300]}")
    check("chat completion reports usage",
          c.get("usage", {}).get("total_tokens", 0) > 0, json.dumps(c.get("usage")))
    print(f"      non-streaming: {dt:.1f}s, content={ch.get('message', {}).get('content')!r}",
          flush=True)

    t0 = time.time()
    code, hdrs, raw = req("/v1/chat/completions", dict(body, stream=True),
                          accept="text/event-stream")
    text = raw.decode("utf-8", "replace")
    lines = [l[5:].strip() for l in text.splitlines() if l.startswith("data:")]
    check("SSE content-type", "text/event-stream" in (hdrs.get("Content-Type") or ""),
          hdrs.get("Content-Type", ""))
    check("SSE terminates with [DONE]", bool(lines) and lines[-1] == "[DONE]",
          text[-200:])
    chunks = [json.loads(l) for l in lines if l != "[DONE]"]
    check("SSE first delta carries the role",
          bool(chunks) and chunks[0]["choices"][0]["delta"].get("role") == "assistant",
          json.dumps(chunks[0]) if chunks else "no chunks")
    check("SSE last chunk carries finish_reason and usage",
          bool(chunks) and chunks[-1]["choices"][0].get("finish_reason") in ("stop", "length")
          and isinstance(chunks[-1].get("usage"), dict),
          json.dumps(chunks[-1]) if chunks else "no chunks")
    streamed = "".join(d["choices"][0]["delta"].get("content") or "" for d in chunks)
    check("SSE streamed text is non-empty", bool(streamed.strip()), repr(streamed[:120]))
    check("SSE object is chat.completion.chunk",
          all(d.get("object") == "chat.completion.chunk" for d in chunks))
    print(f"      streaming: {time.time() - t0:.1f}s, {len(chunks)} chunks, {streamed!r}",
          flush=True)

    code, _, raw = req("/api/generate", {"model": model, "prompt": "Hello", "stream": False,
                                         "options": {"temperature": 0.0, "num_predict": 8}})
    g = json.loads(raw) if code == 200 else {}
    check("POST /api/generate (JSON)",
          code == 200 and g.get("done") is True and isinstance(g.get("response"), str),
          f"{code} {raw[:300]}")

    code, hdrs, raw = req("/api/generate", {"model": model, "prompt": "Hello", "stream": True,
                                            "options": {"temperature": 0.0, "num_predict": 8}})
    nd = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    check("POST /api/generate (NDJSON)",
          code == 200 and bool(nd) and nd[-1].get("done") is True,
          f"{code} {raw[:200]}")

    code, _, raw = req("/v1/chat/completions", dict(body, enable_thinking=False))
    c = json.loads(raw) if code == 200 else {}
    msg = (c.get("choices") or [{}])[0].get("message", {})
    check("enable_thinking=false answers in content",
          code == 200 and bool((msg.get("content") or "").strip())
          and not msg.get("reasoning_content"), f"{code} {raw[:200]}")

    code, _, raw = req("/v1/chat/completions",
                       {"model": model, "messages": msgs, "max_tokens": 48,
                        "temperature": 0.0, "enable_thinking": True})
    c = json.loads(raw) if code == 200 else {}
    msg = (c.get("choices") or [{}])[0].get("message", {})
    check("thinking mode routes reasoning_content",
          code == 200 and bool((msg.get("reasoning_content") or "").strip()),
          f"{code} {raw[:200]}")

    r = urllib.request.Request(BASE + "/v1/chat/completions", data=b"{not json",
                               headers={"Authorization": f"Bearer {KEY}",
                                        "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(r, timeout=10)
        code = 200
    except urllib.error.HTTPError as e:
        code = e.code
    check("malformed JSON is 400", code == 400, str(code))

    code, _, _ = req("/nope")
    check("unknown route is 404", code == 404, str(code))

    # The real contract: drive fennec-core's own client at the running engine.
    try:
        sys.path.insert(0, "D:/Fennec-Suite/fennec-core")
        from fennec_core import engine as eng
        from fennec_core.config import load_config
        from dataclasses import replace
        cfg = replace(load_config(), engine_url=BASE, api_key=KEY)
        client = eng.Engine(cfg)
        check("core client sees the engine running", client.is_running())
        check("core client model_id matches", client.model_id() == model,
              f"{client.model_id()} != {model}")
        msg, usage = client.chat([{"role": "user", "content": "Reply with the single word: ok"}],
                                 max_tokens=24, temperature=0.0, enable_thinking=False)
        check("core client chat() returns an assistant message",
              msg.get("role") == "assistant" and bool((msg.get("content") or "").strip())
              and usage.get("total_tokens", 0) > 0,
              json.dumps([msg, usage])[:300])
        tel = client.telemetry()
        check("core client telemetry() reads all four routes",
              all(k in tel for k in ("health", "metrics", "experts", "profile")),
              json.dumps(tel)[:200])
        pieces = list(client.chat_stream([{"role": "user", "content": "Count to three."}],
                                         max_tokens=24, temperature=0.0,
                                         enable_thinking=False))
        kinds = {p.get("type") for p in pieces if isinstance(p, dict)}
        check("core client chat_stream() yields text and a result",
              "text" in kinds and "result" in kinds, json.dumps(pieces)[:300])
    except Exception as e:
        check("core client integration", False, f"{type(e).__name__}: {e}")

    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed", flush=True)
    for f in FAILS:
        print(f"  FAILED: {f}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())