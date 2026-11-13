"""Golden-reference generator for the Fennec engine.

Every fixture is computed by HuggingFace transformers' own qwen3_5_moe modules,
with the real weights of Qwen3.6-35B-A3B-bf16 loaded in fp32 from safetensors.
The Rust kernels are diffed against these files, so a kernel that disagrees with
transformers is a bug in the Rust, not in the reference.

    pythonw tools/make_goldens.py            # everything
    pythonw tools/make_goldens.py gdn moe    # just those fixtures
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 8)

D: str = r"D:\MEGAsync-sync\colibri-models\Qwen3.6-35B-A3B-bf16"
OUT: str = r"D:\Fennec\tests\fixtures"
os.makedirs(OUT, exist_ok=True)

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeExperts,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextRotaryEmbedding,
    Qwen3_5MoeTopKRouter,
)

CFG = Qwen3_5MoeTextConfig.from_pretrained(D)
INDEX = json.load(open(os.path.join(D, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
HDR: dict[str, dict] = {}
_cache: dict[str, tuple[str, int, int]] = {}


def header(shard: str) -> dict:
    if shard not in _cache:
        with open(os.path.join(D, shard), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        _cache[shard] = h
    return _cache[shard]


def raw(name: str) -> np.ndarray:
    """Read one tensor as a numpy array in its stored dtype (BF16 -> uint16 bits)."""
    shard = INDEX[name]
    h = header(shard)
    item = h[name]
    off = item["data_offsets"]
    with open(os.path.join(D, shard), "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        fh.seek(8 + hl + off[0])
        buf = fh.read(off[1] - off[0])
    dt = {"BF16": np.uint16, "F32": np.float32, "F16": np.uint16, "F64": np.float64}[item["dtype"]]
    return np.frombuffer(buf, dtype=dt).reshape(item["shape"])


def t(name: str) -> torch.Tensor:
    """Tensor in fp32."""
    a = raw(name)
    if a.dtype == np.uint16:  # bf16 bits
        return torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32).copy())
    return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32)).clone()


def fq4(w: torch.Tensor, group: int = 32) -> torch.Tensor:
    """The FQ4 round trip in torch: one f32 scale per `group` columns of each
    row, symmetric around zero, ties to even. Byte-identical nibbles to the
    Rust quantiser (tests/golden.rs::int4_packing_is_byte_identical_to_rabbit),
    so a reference passed through this sees exactly Fennec's weights."""
    *lead, rows, cols = w.shape
    qmax = (1 << 3) - 1
    g = w.reshape(*lead, rows, cols // group, group).float()
    scale = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / qmax, min=1e-8)
    q = torch.clamp(torch.round(g / scale), -8, qmax)
    return (q * scale).reshape(w.shape)


def load_into(mod: torch.nn.Module, prefix: str, strip: str) -> None:
    """Copy checkpoint tensors into a freshly built module, dropping `strip`."""
    want = {k for k in mod.state_dict()}
    pairs = []
    for name, shard in INDEX.items():
        if not name.startswith(prefix):
            continue
        key = name[len(prefix):].lstrip(".")
        if key in want:
            pairs.append((key, name))
    missing = want - {k for k, _ in pairs}
    if missing:
        raise SystemExit(f"no checkpoint tensor for {sorted(missing)[:4]} under {prefix}")
    sd = {k: t(n).reshape(mod.state_dict()[k].shape) for k, n in pairs}
    mod.load_state_dict(sd, strict=True)
    for p in mod.parameters():
        p.data = p.data.float()


def save(name: str, arr) -> None:
    a = arr.detach().cpu().double().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    # a transposed torch view (anything ending in .transpose(1, 2)) reaches numpy
    # with fortran strides, and np.save then writes fortran_order=True, which the
    # Rust reader refuses rather than silently mis-indexing.
    a = np.ascontiguousarray(a)
    np.save(os.path.join(OUT, f"{name}.npy"), a)
    MANIFEST[name] = {"shape": list(a.shape), "bytes": int(a.nbytes)}


MANIFEST: dict[str, dict] = {}
SEED = 20261112
NOTE: dict[str, str] = {}


def rng(*shape, scale=0.02) -> torch.Tensor:
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(*shape, generator=g, dtype=torch.float32) * scale


def main() -> int:
    only = set(sys.argv[1:])
    todo = lambda k: not only or k in only
    t0 = time.time()
    H = CFG.hidden_size

    if todo("rmsnorm"):
        m = Qwen3_5MoeRMSNorm(H, eps=CFG.rms_norm_eps)
        load_into(m, "model.language_model.layers.0.input_layernorm", "input_layernorm")
        x = rng(2, H)
        save("rmsnorm.x", x)
        save("rmsnorm.weight", m.weight)  # zero-centered: out uses (1 + w)
        save("rmsnorm.out", m(x[None]))
        save("rmsnorm.gated_weight", torch.ones(1))
        g = Qwen3_5MoeRMSNormGated(CFG.linear_value_head_dim, eps=CFG.rms_norm_eps)
        load_into(g, "model.language_model.layers.0.linear_attn.norm", "norm")
        hv = CFG.linear_value_head_dim
        save("rmsnorm_gated.x", rng(3, hv))
        save("rmsnorm_gated.gate", rng(3, hv))
        save("rmsnorm_gated.weight", g.weight)
        save("rmsnorm_gated.out", g(rng(3, hv), rng(3, hv)))
        del m, g

    if todo("rope"):
        emb = Qwen3_5MoeTextRotaryEmbedding(CFG)
        pos = torch.arange(8)[None]
        cos, sin = emb(torch.zeros(1, 8, H), pos)
        save("rope.cos", cos)  # [1, 8, rotary_dim]
        save("rope.sin", sin)
        NOTE["rope"] = {"text_positions": "position_ids [B,S] broadcast to 3 grids", "rotary_dim": int(cos.shape[-1])}
        del emb

    if todo("router"):
        r = Qwen3_5MoeTopKRouter(CFG)
        load_into(r, "model.language_model.layers.0.mlp.gate", "gate")
        x = rng(4, H)
        logits, w, idx = r(x[None])
        save("router.x", x)
        save("router.weight", r.weight)
        save("router.logits", logits)
        save("router.weights", w)  # renormalised over the top k
        save("router.indices", idx)
        del r

    if todo("moe"):
        blk = Qwen3_5MoeSparseMoeBlock(CFG)
        load_into(blk, "model.language_model.layers.0.mlp", "mlp")
        x = rng(3, 4, H)
        out = blk(x)
        save("moe.x", x)
        save("moe.out", out)
        for nm in ("gate.weight", "shared_expert_gate.weight",
                   "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
                   "shared_expert.down_proj.weight"):
            save(f"moe.w.{nm}", blk.state_dict()[nm])
        _, w, idx = blk.gate(x.reshape(-1, H))
        save("moe.routing_weights", w)
        save("moe.routing_indices", idx)
        # one routed expert, step by step, so the int4 kernel can be diffed alone
        e = int(idx.reshape(-1, CFG.num_experts_per_tok)[0, 0])
        xe = x.reshape(-1, H)[0:1]
        ex = blk.experts
        gu = torch.nn.functional.linear(xe, ex.gate_up_proj[e])
        gate, up = gu.chunk(2, dim=-1)
        hh = ex.act_fn(gate) * up
        ye = torch.nn.functional.linear(hh, ex.down_proj[e])
        save(f"moe.expert.id", torch.tensor([e]))
        save("moe.expert.gate_up_proj", ex.gate_up_proj[e])
        save("moe.expert.down_proj", ex.down_proj[e])
        save("moe.expert.gate", gate)
        save("moe.expert.up", up)
        save("moe.expert.hh", hh)
        save("moe.expert.y", ye)
        # shared expert on the same row
        se = blk.shared_expert
        g2, u2 = se.gate_proj(xe), se.up_proj(xe)
        save("moe.shared.gate", g2)
        save("moe.shared.up", u2)
        save("moe.shared.hh", se.act_fn(g2) * u2)
        save("moe.shared.y", se.down_proj(se.act_fn(g2) * u2))
        save("moe.shared.gate_logit", blk.shared_expert_gate(xe))
        del blk

    if todo("gdn"):
        m = Qwen3_5MoeGatedDeltaNet(CFG, 0)
        load_into(m, "model.language_model.layers.0.linear_attn", "linear_attn")
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=CFG)
        x = rng(1, 8, H)
        save("gdn.x", x)
        out = m(x, cache_params=cache)
        save("gdn.out_prefill", out)
        save("gdn.conv_state", cache.get_conv_state(0, 0))
        save("gdn.recurrent_state", cache.get_recurrent_state(0, 0))
        # one decode step continuing from that state
        x1 = rng(1, 1, H)
        save("gdn.x_decode", x1)
        out1 = m(x1, cache_params=cache)
        save("gdn.out_decode", out1)
        save("gdn.recurrent_state2", cache.get_recurrent_state(0, 0))
        # the same 8 tokens fed one at a time from an empty cache: this walks
        # fused_recurrent_gated_delta_rule, which is the algorithm Fennec
        # implements, so the comparison is recurrent-vs-recurrent and exact
        cache2 = DynamicCache(config=CFG)
        step_out, step_state = [], []
        for i in range(x.shape[1]):
            step_out.append(m(x[:, i : i + 1], cache_params=cache2))
            step_state.append(cache2.get_recurrent_state(0, 0).clone())
        save("gdn.step_out", torch.cat(step_out, dim=1))
        save("gdn.step_state", torch.stack(step_state))
        # expose the internals for kernel-level diffs
        mixed = m.in_proj_qkv(x).transpose(1, 2)
        conv_raw = m.conv1d(mixed)[:, :, : x.shape[1]]
        conv = torch.nn.functional.silu(conv_raw)
        q, k, v = torch.split(conv, [m.key_dim, m.key_dim, m.value_dim], dim=1)
        save("gdn.mixed_qkv", mixed)
        save("gdn.conv_raw", conv_raw)
        save("gdn.conv_out", conv)
        save("gdn.q", q.transpose(1, 2))
        save("gdn.k", k.transpose(1, 2))
        save("gdn.v", v.transpose(1, 2))
        save("gdn.beta", m.in_proj_b(x).sigmoid())
        save("gdn.g", -m.A_log.float().exp() * torch.nn.functional.softplus(m.in_proj_a(x).float() + m.dt_bias))
        NOTE["gdn"] = {
            "out_prefill": "chunked path, mathematically equal to the recursion but not bit-equal",
            "step_out": "8 x len-1 calls, the recurrent path Fennec implements",
        }
        del m, cache, cache2

    if todo("attn"):
        m = Qwen3_5MoeAttention(CFG, 3)
        load_into(m, "model.language_model.layers.3.self_attn", "self_attn")
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=CFG)
        emb = Qwen3_5MoeTextRotaryEmbedding(CFG)
        x = rng(1, 8, H)
        pos = torch.arange(8)[None]
        pe = emb(x, pos)
        save("attn.x", x)
        # a standalone attention module does not build a causal mask for itself:
        # with attention_mask=None it attends to all 8 positions. Feeding the
        # additive triangular mask is what the full model does while decoding.
        s = x.shape[1]
        causal = torch.zeros(1, 1, s, s)
        causal = causal.masked_fill(
            torch.ones(s, s).tril() == 0, torch.finfo(torch.float32).min
        )
        out, _ = m(x, position_embeddings=pe, attention_mask=causal, past_key_values=cache)
        save("attn.out", out)
        save("attn.gate_logit_scale", torch.tensor([float(m.scaling)]))
        save("attn.q_norm_weight", m.q_norm.weight)
        save("attn.k_norm_weight", m.k_norm.weight)
        hs = x.shape[:-1]
        q, gate = torch.chunk(m.q_proj(x).view(*hs, -1, m.head_dim * 2), 2, dim=-1)
        save("attn.gate", gate.reshape(*hs, -1))
        save("attn.q_pre_rope", m.q_norm(q.view(*hs, -1, m.head_dim)).transpose(1, 2))
        save("attn.k_pre_rope", m.k_norm(m.k_proj(x).view(*hs, -1, m.head_dim)).transpose(1, 2))
        save("attn.v", m.v_proj(x).view(*hs, -1, m.head_dim).transpose(1, 2))
        NOTE["attn"] = {"mask": "additive lower-triangular (causal)", "scaling": float(m.scaling)}
        del m, cache, emb

    if todo("int4"):
        # rabbit's grouped int4: low nibble first, value+8, one f32 scale per group of
        # `gs` columns along each row. Dumped bit-exactly so the Rust can be diffed.
        gu = t("model.language_model.layers.0.mlp.experts.gate_up_proj").reshape(CFG.num_experts, -1, H)[0]
        gs, rows = 32, 8
        w = gu[:rows]
        qmax = (1 << 3) - 1
        groups = w.reshape(rows, H // gs, gs)
        amax = groups.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(amax / qmax, min=1e-8)
        q = torch.clamp(torch.round(groups / scale), -8, qmax).to(torch.int8)
        packed = (q[..., 1::2] + 8) << 4 | (q[..., 0::2] + 8)          # low nibble = even column
        save("int4.src", w)
        save("int4.packed", packed.reshape(rows, H // 2).to(torch.uint8).numpy().astype(np.int64))
        save("int4.scale", scale.squeeze(-1).numpy())
        save("int4.dequant", (q.float() * scale).reshape(rows, H).numpy())
        save("int4.group_size", np.array([gs], np.int64))

    if todo("lm"):
        save("lm.head_weight_slice", t("lm_head.weight")[:8])
        h = t("model.language_model.embed_tokens.weight")[248044]
        save("lm.hidden", h)
        save("lm.logits_top", (h[None] @ t("lm_head.weight").T)[0].topk(8).values)
        save("lm.logits_idx", (h[None] @ t("lm_head.weight").T)[0].topk(8).indices)

    if todo("fwd"):
        # A real prompt through the first four decoder layers in fp32, one layer
        # at a time: 40 layers of fp32 weights is 140 GB, which no machine here
        # has, and the graph Fennec has to reproduce is the same shape either
        # way. Layer 3 is the first full_attention block, so both mixer kinds
        # are covered, and the stack ends at the lm_head.
        #
        # Three streams of the same eight tokens:
        #   h   - prefilled in one call, checkpoint experts.
        #   qh  - prefilled, experts pushed through the same FQ4 round trip
        #         Fennec reads, so identical numbers on both sides.
        #   qs  - one token at a time with a cache, FQ4 experts. This is the only
        #         stream that walks the same *algorithm* Fennec does: seq_len=1
        #         takes torch_recurrent_gated_delta_rule and
        #         causal_conv1d_update, while a prefill takes the chunked solver
        #         that inverts a triangular system. Same math, different rounding.
        from transformers.cache_utils import DynamicCache

        n_layers = int(os.environ.get("FENNEC_FWD_LAYERS", "4"))
        g = torch.Generator().manual_seed(SEED)
        ids = torch.randint(0, 100_000, (8,), generator=g)
        save("fwd.ids", ids.numpy())
        save("fwd.n_layers", np.array([n_layers], np.int64))
        e = raw("model.language_model.embed_tokens.weight")
        h0 = torch.from_numpy((e[ids.numpy()].astype(np.uint32) << 16).view(np.float32).copy())[None]
        save("fwd.embed", h0)
        rope = Qwen3_5MoeTextRotaryEmbedding(CFG)
        pe = rope(h0, torch.arange(h0.shape[1])[None])
        s = h0.shape[1]
        causal = torch.zeros(1, 1, s, s)
        causal = causal.masked_fill(torch.ones(s, s).tril() == 0, torch.finfo(torch.float32).min)

        def build(l: int, round4: bool) -> torch.nn.Module:
            m = Qwen3_5MoeDecoderLayer(CFG, l)
            load_into(m, f"model.language_model.layers.{l}", "")
            if round4:
                sd = m.state_dict()
                for k in list(sd):
                    if k.endswith(("experts.gate_up_proj", "experts.down_proj")):
                        sd[k] = fq4(sd[k], 32)
                m.load_state_dict(sd, strict=True)
                del sd
            return m

        def at(x: torch.Tensor, t: int) -> torch.Tensor:
            """Position `t` out of a [B, S, D] or [S, D] tensor, keeping the rank."""
            return x[:, t : t + 1] if x.dim() == 3 else x[t : t + 1]

        streams = {}
        for tag, round4, stepped in (("h", False, False), ("qh", True, False), ("qs", True, True)):
            cache = DynamicCache(config=CFG)
            prev = None
            for l in range(n_layers):
                m = build(l, round4)
                # the model builds two masks and picks by layer type: the additive
                # 4-D causal one only reaches full_attention, while linear_attention
                # gets the 2-D padding mask, which is None for an unpadded prompt
                full = CFG.layer_types[l] == "full_attention"
                # the two sub-block outputs and the routing decision, so a Rust
                # mismatch says which of the three it lives in
                caps = {"attn": [], "moe": [], "rid": [], "rw": [], "sh": [], "ex": []}

                def mix_hook(_m, _i, out):
                    o = out[0] if isinstance(out, tuple) else out
                    caps["attn"].append(o[:, -1:, :])

                def mlp_hook(_m, _i, out):
                    caps["moe"].append(out[:, -1:, :])

                def flat_hook(key):
                    def hook(_m, _i, out):
                        # the two expert paths run on the flattened [B*S, H] view
                        caps[key].append(out.detach()[-1:].reshape(1, 1, -1))

                    return hook

                def gate_hook(_m, _i, out):
                    # the router flattens to [B*S, k] before it selects
                    caps["rw"].append(out[1][-1:].reshape(1, 1, -1))
                    caps["rid"].append(out[2][-1:].reshape(1, 1, -1))

                handles = [(getattr(m, "self_attn" if full else "linear_attn"), mix_hook),
                           (m.mlp, mlp_hook), (m.mlp.gate, gate_hook),
                           (m.mlp.shared_expert, flat_hook("sh")),
                           (m.mlp.experts, flat_hook("ex"))]
                handles = [(mod, mod.register_forward_hook(fn)) for mod, fn in handles]
                if stepped:
                    # one layer over all positions at a time: layer l reads layer
                    # l-1's outputs, so this is the same chain as a decode loop
                    # without holding four fp32 layers in memory at once
                    out = []
                    exp = {"hh": [], "y": []}
                    for t in range(s):
                        src = h0[:, t : t + 1] if prev is None else prev[t]
                        out.append(
                            m(
                                src,
                                position_embeddings=(at(pe[0], t), at(pe[1], t)),
                                attention_mask=torch.zeros(1, 1, 1, t + 1) if full else None,
                                past_key_values=cache,
                            )[:, -1:, :]
                        )
                        if l == 0 and t == 0:
                            # the eight experts token 0 routes to, each on its own,
                            # so a Rust mismatch names the matrix it got wrong
                            nl = m.post_attention_layernorm(src + caps["attn"][-1])
                            for e in caps["rid"][-1][0, 0].tolist():
                                gate, up = (nl @ m.mlp.experts.gate_up_proj[e].T).chunk(2, dim=-1)
                                hh = m.mlp.experts.act_fn(gate) * up
                                exp["hh"].append(hh)
                                exp["y"].append(hh @ m.mlp.experts.down_proj[e].T)
                            # and a slice of the quantised weights themselves, so a
                            # wrong `hh` can be told apart from a wrong dot product
                            e0 = caps["rid"][-1][0, 0].tolist()[0]
                            mi = CFG.moe_intermediate_size
                            rows = torch.tensor(sorted(set(range(0, 2 * mi, 97)) | {146, 658}))
                            exp["gu"] = m.mlp.experts.gate_up_proj[e0][rows]
                            exp["dn"] = m.mlp.experts.down_proj[e0][rows % H]
                            exp["rows"] = rows
                            exp["id"] = np.array([e0], np.int64)
                    assert torch.cat(out, dim=1).shape == h0.shape
                    save(f"fwd.{tag}{l}", torch.cat(out, dim=1))
                    prev = out
                else:
                    cur = m(
                        h0 if prev is None else prev,
                        position_embeddings=pe,
                        attention_mask=causal if full else None,
                        past_key_values=cache,
                    )
                    save(f"fwd.{tag}{l}", cur)
                    prev = cur
                for h in handles:
                    h[1].remove()
                if stepped:
                    save(f"fwd.qs{l}.attn", torch.cat(caps["attn"], dim=1))
                    save(f"fwd.qs{l}.moe", torch.cat(caps["moe"], dim=1))
                    save(f"fwd.qs{l}.shared", torch.cat(caps["sh"], dim=1))
                    save(f"fwd.qs{l}.experts", torch.cat(caps["ex"], dim=1))
                    save(f"fwd.qs{l}.route_id", torch.cat(caps["rid"], dim=1))
                    save(f"fwd.qs{l}.route_w", torch.cat(caps["rw"], dim=1))
                    if l == 0:
                        save("fwd.qs0.exp_hh", torch.cat(exp["hh"], dim=-1).reshape(1, -1))
                        save("fwd.qs0.exp_y", torch.cat(exp["y"], dim=-1).reshape(1, -1))
                        save("fwd.qs0.gu_probe", exp["gu"])
                        save("fwd.qs0.dn_probe", exp["dn"])
                        save("fwd.qs0.probe_rows", exp["rows"].numpy())
                        save("fwd.qs0.probe_id", exp["id"])
                del m, caps, handles
            streams[tag] = torch.cat(prev, dim=1) if stepped else prev
        nm = Qwen3_5MoeRMSNorm(CFG.hidden_size, eps=CFG.rms_norm_eps)
        load_into(nm, "model.language_model.norm", "")
        hw = raw("lm_head.weight")

        def head_top(x):
            """Logits for the last position, converting the 1 GiB head in row
            blocks so it never needs an fp32 copy of a matrix read once."""
            v = x[0, -1].float().numpy()
            lg = np.empty(hw.shape[0], np.float32)
            for r0 in range(0, hw.shape[0], 4096):
                blk = (hw[r0 : r0 + 4096].astype(np.uint32) << 16).view(np.float32)
                lg[r0 : r0 + blk.shape[0]] = blk @ v
            return lg

        qn = nm(streams["qs"])
        save("fwd.norm", qn)
        lgq = head_top(qn)
        top = np.argsort(-lgq)[:8]
        save("fwd.logits_idx", top)
        save("fwd.logits_top", lgq[top])
        # the prefill streams are the controls: what the format and the choice of
        # solver each cost on their own
        n32 = nm(streams["h"])
        save("fwd.norm32", n32)
        lg32 = head_top(n32)
        save("fwd.logits_top32", lg32[top])
        NOTE["fwd"] = {
            "h{l}": "residual stream, 8-token prefill, fp32 experts",
            "qh{l}": "same prefill with the experts pushed through FQ4",
            "qs{l}": "token-at-a-time with a cache, FQ4 experts: Fennec's own algorithm",
            "qs{l}.attn": "that stream's mixer output, before the residual add",
            "qs{l}.moe": "and its MoE block output (routed + gated shared expert)",
            "qs{l}.shared": "the shared expert before its sigmoid gate",
            "qs{l}.experts": "the top-k routed experts after their weights, before the add",
            "qs{l}.route_id": "[seq, top_k] expert ids, qs{l}.route_w the renormalised weights",
            "logits_idx": "top-8 ids of the stepped stream's last position",
            "logits_top32": "those same 8 ids read off the fp32 prefill",
        }
        del cache, nm, rope

    json.dump({"seed": SEED, "model": D, "seconds": round(time.time() - t0, 2),
               "hidden_size": H, "fixtures": MANIFEST, "notes": NOTE},
              open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"), indent=1)
    print("goldens ok", len(MANIFEST), "fixtures in", round(time.time() - t0, 1), "s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
