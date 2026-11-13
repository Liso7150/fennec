"""Golden-reference generator for Fennec's dense Qwen3 lane.

Every fixture is computed by HuggingFace transformers' own qwen3 modules with
the real weights of the checkpoint loaded in fp32, so a Rust mismatch names a
bug in the Rust, not in the reference. The MoE generator
(`make_goldens.py`) is the same shape; this one has no router, no experts and
no gated-delta-net to instrument.

    python tools/make_goldens_qwen3.py                     # everything
    python tools/make_goldens_qwen3.py --model D:/path/to # another dense export

Checkpoints of this family are small enough that fp32 for the whole model is
not a decision, so the reference walks the real modules directly.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 8)

DEFAULT_MODEL = r"D:\Fennec-Suite\downloads\models\MNN-Pocket-0.3B"
OUT = r"D:\Fennec\tests\fixtures"
os.makedirs(OUT, exist_ok=True)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MANIFEST: dict[str, dict] = {}
NOTE: dict[str, str] = {}
SEED = 20261113
# the same eight-token sentence for every run: short enough to step one token
# at a time in fp32, long enough that position 7 exercises a real KV history
PROMPT = "The capital of Poland is Warsaw, and"


def save(name: str, arr) -> None:
    a = arr.detach().cpu().double().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    # a transposed torch view reaches numpy with fortran strides, and np.save
    # then writes fortran_order=True, which the Rust reader refuses
    a = np.ascontiguousarray(a)
    np.save(os.path.join(OUT, f"{name}.npy"), a)
    MANIFEST[name] = {"shape": list(a.shape), "bytes": int(a.nbytes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FENNEC_DENSE_MODEL", DEFAULT_MODEL))
    args = ap.parse_args()
    d = args.model
    cfg = AutoConfig.from_pretrained(d)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(d)
    ids = tok(PROMPT, add_special_tokens=False)["input_ids"][:8]
    h = cfg.hidden_size
    print(f"{cfg.model_type}: {cfg.num_hidden_layers} layers, hidden {h}, "
          f"intermediate {cfg.intermediate_size}, heads {cfg.num_attention_heads}/"
          f"{cfg.num_key_value_heads} x {cfg.head_dim}, tied {cfg.tie_word_embeddings}")

    save("q3.ids", np.asarray(ids, np.int64))
    emb = model.get_input_embeddings()(torch.tensor([ids]))
    save("q3.embed", emb)

    text_model = model.model
    rotary = getattr(text_model, "rotary_emb", None) or getattr(text_model, "rotary_embed")
    norm = text_model.norm
    x = emb
    pos = torch.arange(len(ids)).unsqueeze(0)
    cos, sin = rotary(x, pos)
    save("q3.rope_cos", cos)
    save("q3.rope_sin", sin)

    # Every stage of a decoder block, captured from the module that computes it.
    # A single `self_attn` output is not enough to debug a mismatch: the block
    # is norm -> three projections -> per-head norms -> rope -> cache -> softmax
    # -> output projection, and a diff that names the stage saves a bisect.
    # Each entry is (attribute path on the layer, fixture key, pre-hook?).
    PIECES = [
        ("input_layernorm", "n", False),
        ("self_attn.q_proj", "q", False),
        ("self_attn.q_norm", "qn", False),
        ("self_attn.k_proj", "k", False),
        ("self_attn.k_norm", "kn", False),
        ("self_attn.v_proj", "v", False),
        ("self_attn.o_proj", "ctx", True),
        ("self_attn", "a", False),
        ("post_attention_layernorm", "np", False),
        ("mlp.gate_proj", "g", False),
        ("mlp.up_proj", "u", False),
        ("mlp.down_proj", "act", True),
        ("mlp", "m", False),
    ]

    def module_of(layer, path):
        m = layer
        for part in path.split("."):
            m = getattr(m, part)
        return m

    def last(o):
        o = o[0] if isinstance(o, tuple) else o
        return o[:, -1:, :].clone()

    # token-at-a-time with a cache: exactly the algorithm the engine runs, so
    # every position of the fixture is a decode step and nothing is batched.
    # The step-0 states are the ones a single-token decode produces, and they
    # are the only ones worth a file per stage: every later step is covered by
    # the stream fixtures.
    cache = DynamicCache()
    prev = []
    logits = []
    for t in range(len(ids)):
        step = x[:, t:t + 1]
        for l, layer in enumerate(text_model.layers):
            got: dict[str, torch.Tensor] = {}
            handles = []
            if t == 0:
                for path, key, pre in PIECES:
                    m = module_of(layer, path)
                    if pre:
                        handles.append(
                            m.register_forward_pre_hook(
                                lambda _mod, _in, k=key: got.__setitem__(k, last(_in))
                            )
                        )
                    else:
                        handles.append(
                            m.register_forward_hook(
                                lambda _mod, _in, out, k=key: got.__setitem__(k, last(out))
                            )
                        )
            step = layer(
                step,
                position_embeddings=(cos[:, t:t + 1], sin[:, t:t + 1]),
                attention_mask=torch.zeros(1, 1, 1, t + 1),
                past_key_values=cache,
            )
            for hdl in handles:
                hdl.remove()
            save(f"q3.qs{t}.{l}", step)
            if t == 0:
                for _path, key, _pre in PIECES:
                    save(f"q3.{key}0.{l}", got[key])
        prev.append(step)
        logits.append(model.lm_head(norm(step)))
    save("q3.stream", torch.cat(prev, dim=1))
    # one token per step, so each entry is [1, 1, vocab] and the steps stack on
    # axis 1; concatenating on axis 2 would silently write one long row instead
    save("q3.logits", torch.cat(logits, dim=1))

    lg = logits[-1][0, -1]
    top = torch.argsort(lg, descending=True)[:8]
    save("q3.logits_idx", top.numpy())
    save("q3.logits_top", lg[top].numpy())
    # the tokenizer's pieces carry U+0120 word-boundar marks, which a cp1250
    # console refuses, so the summary prints escapes rather than the glyphs
    print("  next-token top 5:", [(int(i), ascii(tok.convert_ids_to_tokens(int(i)))) for i in top[:5]])
    NOTE["q3"] = {
        "ids": "the eight prompt token ids",
        "embed": "input embeddings for them",
        "rope_cos/sin": "rotary tables at positions 0..7 (full width)",
        f"qs<t>.<l>": f"hidden state after layer l at decode step t, [1, 1, {h}]",
        "a0.<l>/m0.<l>": f"layer {h} mixer / MLP output at step 0, before the residual add",
        "stream": "all eight decode steps concatenated",
        "logits": f"[1, 8, {cfg.vocab_size}] logits, one row per step",
        "logits_idx/top": "the last step's top-8 ids and their logits",
    }
    json.dump({"seed": SEED, "model": d, "prompt": PROMPT,
               "seconds": round(time.time() - t0, 2), "hidden_size": h,
               "fixtures": MANIFEST, "notes": NOTE},
              open(os.path.join(OUT, "manifest-qwen3.json"), "w", encoding="utf-8"), indent=1)
    print(f"goldens ok {len(MANIFEST)} fixtures in {round(time.time() - t0, 1)} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
