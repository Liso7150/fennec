#!/usr/bin/env python3
"""Draw the symmetric serve-protocol matrix from the measured JSON. Stdlib only.

Every live cell was measured the same way: `serve` plus one streaming chat
completion, same prompt, same token budget, median of N pairs. Cells that cannot
run on this box are not drawn as zero and not drawn as a number -- they are
listed in the README with the evidence for why they are unavailable.

    python tools/matrix_graph.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SUITE = Path("D:/Fennec-Suite")
sys.path.insert(0, str(HERE / "tools"))

from engine_graph import bars  # noqa: E402

SRC = SUITE / "benchmarks" / "results-backend-matrix.json"
OUT = HERE / "docs"

LABEL = {
    "colibri-cpu": ("Colibri (CPU)", "JustVugg/colibri", "CPU only"),
    "colibri-vulkan": ("Colibri (Vulkan)", "JustVugg/colibri", "CPU + GPU"),
    "fennec-cpu": ("Fennec 2.5.0 (CPU)", "Liso7150/fennec", "CPU only"),
    "fennec-vulkan": ("Fennec 2.5.0 (Vulkan)", "Liso7150/fennec", "CPU + GPU"),
    "fennec-cuda": ("Fennec 2.5.0 (CUDA)", "Liso7150/fennec", "CPU + GPU"),
}


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = []
    for cell in data["cells"]:
        if cell.get("status") != "ok":
            continue
        name, repo, tier = LABEL.get(cell["cell"], (cell["cell"], "", ""))
        rows.append({
            "name": name, "repo": repo, "tier": tier,
            "tok_s": cell["median_tokens_per_s"],
            "ttft_ms": cell["median_ttft_s"] * 1000.0,
            "tpot_ms": cell["median_tpot_ms"],
            "total_ms": cell["median_total_s"] * 1000.0,
            "load_s": cell["startup_s"],
        })
    if not rows:
        raise SystemExit("no ok cells in the matrix JSON")
    rows.sort(key=lambda r: -r["tok_s"])

    sub = (f"One protocol for both engines: serve + one streaming completion, "
           f"{data['tokens']} tokens, ctx {data['ctx']}, median of {data['pairs']}, "
           f"measured {data['measured_at']}. Token counts come from "
           f"usage.completion_tokens, not from counting SSE chunks.")
    charts = [
        ("engine-matrix-tok.svg", "tok_s", "tokens / second, higher is better",
         "Serve throughput — same checkpoint, same protocol, both engines", sub,
         lambda v: f"{v:.2f}"),
        ("engine-matrix-ttft.svg", "ttft_ms", "milliseconds, lower is better",
         "Time to first token — over HTTP, both engines", sub,
         lambda v: f"{v / 1000:.2f} s"),
        ("engine-matrix-tpot.svg", "tpot_ms", "milliseconds, lower is better",
         "Time per output token — decode window only", sub,
         lambda v: f"{v:.1f} ms"),
        ("engine-matrix-total.svg", "total_ms", "milliseconds, lower is better",
         "Total request latency — measured wall time", sub,
         lambda v: f"{v / 1000:.2f} s"),
        ("engine-matrix-load.svg", "load_s", "seconds to ready, lower is better",
         "Cold start — spawn until /v1/models answers", sub,
         lambda v: f"{v:.1f} s"),
    ]
    for name, key, unit, title, subtitle, fmt in charts:
        (OUT / name).write_text(
            bars(rows, key, unit, title, subtitle, fmt=fmt,
                 lower_is_better=(key != "tok_s")),
            encoding="utf-8")
    print(f"wrote {len(charts)} charts to {OUT}")
    for r in rows:
        print(f"  {r['name']:26s} {r['tok_s']:7.3f} tok/s  "
              f"ttft {r['ttft_ms'] / 1000:6.3f} s  tpot {r['tpot_ms']:8.3f} ms  "
              f"load {r['load_s']:5.1f} s")


if __name__ == "__main__":
    main()