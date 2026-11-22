#!/usr/bin/env bash
# Rebuild the checked-in SPIR-V from the GLSL sources. The .spv files are committed
# so `cargo build --offline` needs no shader compiler; run this after editing a
# .comp. GLSLANG overrides the compiler path.
#
# No -O, deliberately: it halves the L2-resident gemv part rows but moves the issued
# mixer layer by 1.7% and costs gdn_rec 27% (benchmarks/artifacts/raw/
# probe-2026-11-22-vk-ceiling in the suite). Add it only with a served ABBA pair.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
GLSLANG="${GLSLANG:-/d/Fennec-Suite/engine-bench/glslang/bin/glslang.exe}"
for src in "$here"/src/shaders/*.comp; do
    out="${src%.comp}.spv"
    "$GLSLANG" -V --target-env vulkan1.0 -o "$out" "$src"
    printf '%s -> %s (%s bytes)\n' "$(basename "$src")" "$(basename "$out")" "$(stat -c %s "$out")"
done