#!/usr/bin/env python3
"""Draw docs/engine-benchmark.svg from docs/engine-benchmark.json. Stdlib only.

The numbers in the JSON are measurements, not estimates: each row carries the
log it came from. Run this after adding a run, never edit the SVG by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
INK = "#1f2328"
MUTED = "#656d76"
GRID = "#d8dee4"
WIN = "#1a7f37"
FENNEC = "#bc4c00"
OTHER = "#0969da"
CPU = "#8250df"


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def color(e: dict) -> str:
    if e["repo"] == "Liso7150/fennec":
        return FENNEC
    if e["tier"] == "CPU only":
        return CPU
    return OTHER


def wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def bars(engines: list[dict], key: str, unit: str, title: str, sub: str,
         fmt=lambda v: f"{v:.2f}", lower_is_better: bool = False) -> str:
    width, left, tail, row_h, bar_h = 940, 250, 150, 46, 20
    sub_lines = wrap(sub, 118)
    top = 62 + 16 * len(sub_lines)
    height = top + row_h * len(engines) + 46
    vals = [e.get(key) for e in engines if e.get(key) is not None]
    best = min(vals) if lower_is_better else max(vals) if vals else None
    scale = (width - left - tail) / max(vals) if vals else 0.0
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
         f'role="img" aria-label="{esc(title)}" font-family="-apple-system,Segoe UI,'
         f'Helvetica,Arial,sans-serif">',
         f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
         f'<text x="20" y="30" font-size="17" font-weight="600" fill="{INK}">{esc(title)}</text>']
    for i, line in enumerate(sub_lines):
        s.append(f'<text x="20" y="{52 + i * 16}" font-size="12" fill="{MUTED}">{esc(line)}</text>')
    for i, e in enumerate(engines):
        y = top + i * row_h
        v = e.get(key)
        c = color(e)
        s.append(f'<text x="{left - 12}" y="{y + bar_h / 2 + 4}" text-anchor="end" '
                 f'font-size="13" fill="{INK}">{esc(e["name"])}</text>')
        if v is None:
            s.append(f'<text x="{left + 4}" y="{y + bar_h / 2 + 4}" font-size="12" '
                     f'fill="{MUTED}">not reported</text>')
        else:
            w = max(2.0, v * scale)
            s.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" '
                     f'rx="3" fill="{c}"/>')
            tag = "fastest" if lower_is_better else "winner"
            label = f"{fmt(v)}  {tag}" if v == best else fmt(v)
            s.append(f'<text x="{left + w + 8:.1f}" y="{y + bar_h - 4}" font-size="12" '
                     f'font-weight="{600 if v == best else 400}" '
                     f'fill="{WIN if v == best else INK}">{esc(label)}</text>')
        s.append(f'<text x="{width - 16}" y="{y + bar_h / 2 + 4}" text-anchor="end" '
                 f'font-size="11" fill="{MUTED}">{esc(e["tier"])}</text>')
        s.append(f'<line x1="{left}" y1="{y + row_h - 8}" x2="{width - tail + 110}" '
                 f'y2="{y + row_h - 8}" stroke="{GRID}"/>')
    s.append(f'<text x="{left}" y="{height - 16}" font-size="11" fill="{MUTED}">'
             f'{esc(unit)}</text>')
    s.append("</svg>")
    return "".join(s)


def main() -> None:
    data = json.loads((HERE / "docs" / "engine-benchmark.json").read_text(encoding="utf-8"))
    engines = data["engines"]
    out = HERE / "docs"
    (out / "engine-benchmark.svg").write_text(
        bars(engines, "tok_s", "tokens / second, higher is better",
             "Decode throughput — Qwen3.6-35B-A3B on one RTX 4060 Laptop + Core Ultra 7 155H",
             data["note"]), encoding="utf-8")
    (out / "engine-load.svg").write_text(
        bars(engines, "load_s", "seconds to first ready token, lower is better",
             "Cold start — model load until the engine answers",
             "Same checkpoint, same machine. Fennec borrows the int4 container's packed "
             "nibbles straight out of the mapping instead of copying them into a host "
             "pool, which is what took its cold start from 96.9 s to 11.8 s. fox loads "
             "a pre-quantised GGUF and defers the rest to first touch.",
             fmt=lambda v: f"{v:.1f} s",
             lower_is_better=True),
        encoding="utf-8")
    # The three latency metrics. Total is the user-visible one and is defined as
    # TTFT + TPOT x output tokens, so it is drawn from the same rows rather than
    # timed separately: wall time and that sum must agree.
    latency = [
        ("engine-ttft.svg", "ttft_ms", "Time to first token",
         "Milliseconds from the prompt arriving to the first output token, lower is better. "
         "Fennec walks the prompt one position at a time, so this is the sum of the "
         "prompt-length steps, not a separate prefill pass.",
         lambda v: f"{v / 1000:.2f} s"),
        ("engine-tpot.svg", "tpot_ms", "Time per output token",
         "Milliseconds per token after the first, lower is better. The mean of the "
         "decode steps only; the prompt steps are excluded so TTFT is not counted twice.",
         lambda v: f"{v:.1f} ms"),
        ("engine-latency.svg", "total_latency_ms", "Total latency — TTFT + TPOT x output tokens",
         "Milliseconds for the whole request, lower is better. Computed as "
         "TTFT + (TPOT x output tokens) and cross-checked against measured wall time.",
         lambda v: f"{v / 1000:.2f} s"),
    ]
    for name, key, title, sub, fmt in latency:
        (out / name).write_text(
            bars(engines, key, "milliseconds, lower is better", title, sub,
                 fmt=fmt, lower_is_better=True),
            encoding="utf-8")
    print(f"wrote {out / 'engine-benchmark.svg'}, {out / 'engine-load.svg'} and "
          f"{', '.join(n for n, *_ in latency)}")


if __name__ == "__main__":
    main()