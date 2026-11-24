//! Multi-token speculation from the `mtp.*` block the checkpoint ships with.
//!
//! Qwen3.5/3.6 exports one decoder block whose input is not the previous token's
//! embedding alone but the pair `(embedding(token), trunk hidden)`, trained to
//! predict two tokens ahead. That is a drafter: it is ~1/40th of the trunk's
//! dense read, and the trunk can then check what it proposed for the price of
//! one batched pass instead of one pass per token, because
//! [`crate::model::Model::prefill`] reads every dense weight once for all its
//! rows and is bit-identical to the same rows run one at a time.
//!
//! The round, and where its cost goes:
//!
//! * `k` draft steps: each is one [`Layer`](crate::model::Layer) pass plus one
//!   `lm_head` pass. The head is not small - 248320 x 2048, half a GiB even at
//!   one byte per weight - and a draft needs its own greedy token, so this is
//!   the term that caps `k`. It is why Strata's `--mtp 2` bought 1.53x rather
//!   than the ~2.9x the trunk's amortisation alone would suggest
//!   (`benchmarks/artifacts/hub`, #481).
//! * one verify pass over `k + 1` rows: the trunk reads its mixer and its
//!   expert-union once for the block. The union is not free - 8 picks per token
//!   out of 256 experts overlap slowly - so the trunk's saving grows with `k`
//!   more slowly than its dense half's does.
//! * rollback: attention caches forget by row count, so rows past the commit
//!   point cost nothing to drop. A gated-delta-net state is a running matrix
//!   with no undo, so the trunk's 63 MiB of rec/conv state is snapshotted before
//!   the verify and written back only on a round that rejects.
//!
//! Output is unchanged, not merely similar: every token that leaves this module
//! has been produced by the trunk's own argmax over the real prefix, so the
//! generated sequence is the same one `--mtp 0` generates. That holds only while
//! the acceptance test is the same rule the sequential loop uses - argmax over
//! the row's logits. A sampler with temperature, or a repetition penalty that
//! edits the logits from the sequence's own history, is not a predicate a block
//! can be verified against without the full rejection-sampling correction, so
//! [`Spec`] is gated to greedy-without-penalty at the call sites.
//!
//! Tier gates, and how much each tier's guarantee is worth. The host tier is
//! byte-identical to `--mtp 0` by construction: the verify runs the same host
//! kernels the sequential loop runs, batched - and it is the tier where the
//! feature pays, 91.762 against 116.198 ms/token served ABBA (2026-11-23,
//! `engine-bench/raw/fennec-mtp-attribution.json`). CUDA verifies through the
//! same chunked device prefill the prompt takes - whose projections are the
//! per-row device gemv decode itself launches (the CUDA batch entry is n such
//! calls), and whose gated-delta-net recursion runs on the host copy of the
//! state, so the same [`StateSave`](crate::model::StateSave) rollback applies
//! once the state is synced. The guarantee held: the served greedy A/B against
//! fresh `--mtp 0` processes was byte-identical over 8 visits x 128 tokens
//! (`engine-bench/raw/fennec-mtp-device-lossless.json`). The speed did not:
//! the same ABBA put a verify round at 71.309 ms/token against 59.897
//! sequential (`engine-bench/raw/fennec-mtp-attribution-cuda.json`) - the
//! chunk wrapper, expert union and readback a 3-row verify pays outweigh the
//! drafts it saves even at 3.00 accepted tokens per round - so the serve gate
//! declines every device tier and keeps the drafter on the host tier.

use std::sync::Arc;

use crate::config::{LayerKind, Result, TextConfig, MTP_PREFIX};
use crate::kernel;
use crate::model::{dense_half, norm_w, Dense, Layer, LayerState, Model, StateSave};
use crate::safetensors::Shards;

/// The largest block the driver will ever verify in one pass, and so the number
/// of hidden-row slots it holds.
const MAX_K: usize = 4;

/// First index of the maximum, which is how [`Sample::pick`](crate::serve::Sample)
/// breaks a tie at temperature 0. Speculation is only lossless if the drafter and
/// the verifier agree with the sequential loop down to that.
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

/// The drafter's weights: one decoder block under `mtp.layers.0.*`, the `fc`
/// projection that folds the (embedding, hidden) pair back to `hidden_size`, and
/// the three norms around them. No loader of its own: the block's tensor names
/// match the trunk's layer schema under a different prefix, so
/// [`Layer::with_experts`] reads it as any other layer.
pub struct Mtp {
    pub cfg: Arc<TextConfig>,
    pub layer: Layer,
    /// `[hidden, 2 * hidden]`: the two normalised halves concatenated back down.
    pub fc: Dense,
    pub n_emb: Vec<f32>,
    pub n_hid: Vec<f32>,
    /// `mtp.norm`: what the block's output passes through on its way to the head.
    pub norm: Vec<f32>,
    /// The embedding table the block reads: the trunk's, unless the checkpoint
    /// gives the drafter one of its own.
    pub embed: String,
    /// Which half of the concat the embedding goes in, and where the block's rows
    /// sit in the rope table.
    ///
    /// Neither is stated by the checkpoint, and guessing wrong does not fail - it
    /// silently drafts garbage, which costs speed and never correctness. Both
    /// were settled by draft acceptance instead of by reading a reference
    /// implementation: see the sweep in `tests/mtp.rs`.
    pub embed_first: bool,
    pub pos_off: usize,
    pub threads: usize,
}

/// The drafter's cache: one row per token it has consumed, and the row count its
/// rope positions come from.
pub struct MtpState {
    pub st: Vec<LayerState>,
    pub plen: usize,
}

/// Forget the drafts the trunk rejected: `keep` rows survive, and the rope
/// position counter goes back with them. The two must move together - a cache
/// whose rows run past `plen` leaves the next step appending over a row it still
/// attends to, which is a wrong draft, not a wrong answer. The block is
/// attention-only (`mtp.layers.0.self_attn`), so there is no recurrent state to
/// restore here, and no weights to touch, which is why this is a function of the
/// state rather than a method of [`Mtp`].
pub fn rewind(st: &mut MtpState, keep: usize) {
    if let Some(LayerState::Full(a)) = st.st.first_mut() {
        a.len = keep.min(a.len);
    }
    st.plen = keep;
}

impl Mtp {
    /// `Ok(None)` when the checkpoint carries no drafter - no `mtp` key in the
    /// config, or no block in the shard index, which is the case for every
    /// checkpoint converted to FQ4 before the MTP weights were kept.
    pub fn load(
        sh: &Arc<Shards>,
        trunk: &Arc<TextConfig>,
        embed: &str,
        group: usize,
        threads: usize,
        kv_cap: usize,
    ) -> Result<Option<Mtp>> {
        if trunk.mtp_num_hidden_layers == 0 || trunk.num_hidden_layers == 0 {
            return Ok(None);
        }
        let fc = format!("{MTP_PREFIX}.fc.weight");
        if !sh.contains(&fc) || !sh.contains(&format!("{MTP_PREFIX}.layers.0.self_attn.q_proj.weight")) {
            return Ok(None);
        }
        // The block is a decoder layer of the same family: same width, same
        // experts, same router, same norms. Only its depth, its mixer kind and
        // its prefix differ, and `num_hidden_layers` is what bounds the layer
        // index the loader asks for.
        let mut mc = (**trunk).clone();
        mc.prefix = MTP_PREFIX.to_string();
        mc.num_hidden_layers = 1;
        mc.layer_types = vec![LayerKind::Full];
        mc.mtp_num_hidden_layers = 0;
        let mc = Arc::new(mc);
        let ids: Vec<usize> = (0..trunk.num_experts).collect();
        let layer = Layer::with_experts(sh, mc.clone(), 0, &ids, None, group, threads, kv_cap, false)?;
        let zc = trunk.norm_zero_centered;
        // The shipped block shares the trunk's table
        // (`mtp_use_dedicated_embeddings: false`); one that carries its own
        // names it under its own prefix.
        let own = format!("{MTP_PREFIX}.embed_tokens.weight");
        let table = if trunk.mtp_use_dedicated_embeddings && sh.contains(&own) { &own } else { embed };
        Ok(Some(Mtp {
            fc: dense_half(sh, &fc)?.0,
            n_emb: norm_w(sh, &format!("{MTP_PREFIX}.pre_fc_norm_embedding.weight"), zc)?,
            n_hid: norm_w(sh, &format!("{MTP_PREFIX}.pre_fc_norm_hidden.weight"), zc)?,
            norm: norm_w(sh, &format!("{MTP_PREFIX}.norm.weight"), zc)?,
            layer,
            cfg: mc,
            embed: table.to_string(),
            embed_first: true,
            pos_off: 0,
            threads: threads.max(1),
        }))
    }

    pub fn state(&self, kv_cap: usize) -> MtpState {
        MtpState {
            st: vec![self.layer.state(&self.cfg, kv_cap)],
            plen: 0,
        }
    }

    /// One draft step: consume `tok` paired with `hid`, leave this block's own
    /// hidden in `hid` for the next step (or for the head, via [`Mtp::head_in`]).
    ///
    /// The pairing is the trained one - `hid` is the trunk's pre-final-norm row
    /// for the token *before* `tok` on the first step of a round, and this
    /// block's previous output after that. Chaining its own output is what makes
    /// the second and later drafts weaker than the first, which is exactly the
    /// shape of the acceptance numbers: ~97% at depth 1, and less with depth.
    pub fn step(
        &self,
        sh: &Shards,
        tok: usize,
        hid: &mut [f32],
        st: &mut MtpState,
        scratch: &mut [f32],
    ) -> Result<()> {
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let e = sh.row_f32(&self.embed, tok)?;
        let (a, b) = scratch.split_at_mut(h);
        let (eb, hb) = if self.embed_first { (a, b) } else { (b, a) };
        kernel::rmsnorm(&e, &self.n_emb, eps, eb);
        kernel::rmsnorm(hid, &self.n_hid, eps, hb);
        self.fc.gemv(h, 2 * h, &scratch[..2 * h], hid, self.threads);
        self.layer.forward(hid, &mut st.st[0], st.plen + self.pos_off);
        st.plen += 1;
        Ok(())
    }

    /// The block's hidden as the head wants it.
    pub fn head_in(&self, hid: &[f32], out: &mut [f32]) {
        kernel::rmsnorm(hid, &self.norm, self.cfg.rms_norm_eps as f32, out);
    }
}

/// The round driver, and the buffers it needs, so `serve` and `run` speculate
/// through the same code. It owns the drafter's cache; the trunk's cache stays
/// with the caller, which also owns its stop conditions.
pub struct Spec {
    pub mtp: Arc<Mtp>,
    /// Drafts per round. 2 is the shipped default: at 3 the third draft's
    /// acceptance no longer pays for the extra head pass.
    pub k: usize,
    pub st: MtpState,
    /// Rows the verify pass may feed: the pending token plus `k` drafts.
    toks: Vec<u32>,
    hs: Vec<f32>,
    /// The same rows after the trunk's final norm, which is what the head reads.
    hns: Vec<f32>,
    lg: Vec<f32>,
    hid: Vec<f32>,
    hn: Vec<f32>,
    scratch: Vec<f32>,
    draft: Vec<u32>,
    logits: Vec<f32>,
    slots: Vec<StateSave>,
    /// What the block bought: rounds run, drafts proposed, drafts accepted, and
    /// tokens committed. `accepted / proposed` is the acceptance rate the
    /// arithmetic above is priced with.
    pub rounds: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub committed: usize,
    /// A round that cut its block at the stop token. The sequence ends there, and
    /// the token itself stays out of `ids`, exactly as the sequential loop leaves
    /// it out.
    stopped: bool,
}

impl Spec {
    pub fn new(mtp: Arc<Mtp>, k: usize, kv_cap: usize) -> Spec {
        let k = k.min(MAX_K);
        let (h, v) = (mtp.cfg.hidden_size, mtp.cfg.vocab_size);
        Spec {
            st: mtp.state(kv_cap),
            toks: Vec::with_capacity(k + 1),
            hs: vec![0f32; (k + 1) * h],
            hns: vec![0f32; (k + 1) * h],
            lg: vec![0f32; (k + 1) * v],
            hid: vec![0f32; h],
            hn: vec![0f32; h],
            scratch: vec![0f32; 2 * h],
            draft: Vec::with_capacity(k),
            logits: vec![0f32; v],
            slots: Vec::new(),
            mtp,
            k,
            rounds: 0,
            proposed: 0,
            accepted: 0,
            committed: 0,
            stopped: false,
        }
    }

    /// Whether a round cut its block at the caller's stop token: the turn is over,
    /// and the sequences `--mtp 0` and `--mtp N` end at the same token.
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// One round of draft-verify-commit.
    ///
    /// `ids` ends with the pending token - the trunk sampled it from the row at
    /// `pos - 1` and has not fed it yet - and `hprev` is that row's hidden, which
    /// is what the drafter pairs with the pending token. On return `ids` carries
    /// the committed tokens, `hprev` holds the hidden of the last row the trunk
    /// actually kept, and the return value is how many rows the trunk advanced,
    /// which is the caller's new `pos` delta.
    ///
    /// `stop` is the caller's end-of-turn test, applied to each newly committed
    /// token in order: a stop token inside an accepted block cuts the block before
    /// it, the rows past it roll back like any rejection, and [`Spec::stopped`]
    /// then tells the caller not to run another round. A round can therefore
    /// commit 0 tokens.
    pub fn round(
        &mut self,
        model: &Model,
        sh: &Shards,
        ids: &mut Vec<u32>,
        st: &mut [LayerState],
        hprev: &mut [f32],
        pos: usize,
        budget: usize,
        stop: &dyn Fn(u32) -> bool,
    ) -> Result<usize> {
        let (h, v) = (model.cfg.hidden_size, model.cfg.vocab_size);
        let k = self.k.min(budget.saturating_sub(1));
        let nr = k + 1;

        // 1. Draft. Each step pays one head pass for its own greedy token.
        self.draft.clear();
        self.hid.copy_from_slice(hprev);
        let base = self.st.plen;
        for j in 0..k {
            let tok = if j == 0 {
                *ids.last().ok_or("spec: no pending token")? as usize
            } else {
                self.draft[j - 1] as usize
            };
            self.mtp.step(sh, tok, &mut self.hid, &mut self.st, &mut self.scratch)?;
            self.mtp.head_in(&self.hid, &mut self.hn);
            model.logits(sh, &self.hn, &mut self.logits)?;
            self.draft.push(argmax(&self.logits) as u32);
        }

        // 2. Verify: the pending token and all `k` drafts, one batched pass.
        self.toks.clear();
        self.toks.push(*ids.last().ok_or("spec: no pending token")?);
        self.toks.extend_from_slice(&self.draft);
        for (i, &t) in self.toks.iter().enumerate() {
            let row = model.embed_row(sh, t as usize)?;
            self.hs[i * h..(i + 1) * h].copy_from_slice(&row);
        }
        // The device tier's chunk recursion runs on the host copy of the state,
        // so the snapshot has to be taken after that copy is made real:
        // `sync_host_state` pulls what decode left in VRAM back first, and is a
        // no-op for a state that is already host-authoritative, which is what a
        // rejected round leaves behind.
        let device = model.prefill_device_ready();
        if device {
            model.sync_host_state(st)?;
        }
        if self.slots.len() != st.len() {
            self.slots = StateSave::all(st);
        } else {
            StateSave::refresh(&mut self.slots, st);
        }
        if device {
            model.prefill_device(&mut self.hs[..nr * h], st, pos)?;
        } else {
            model.prefill(&mut self.hs[..nr * h], st, pos);
        }
        // The head reads what the final norm left, not the trunk's raw row: the
        // norm's per-column gain is not a constant, so a row fed to the head
        // unnormalised has a different argmax than the same row after it. The
        // sequential loop norms before every head pass (see `Server::generate`),
        // and the acceptance test is only the same rule as the sequential one if
        // this pass applies it too. Skipping it cost 13 rejected drafts and one
        // divergent answer in 64 tokens of prose.
        let eps = model.cfg.rms_norm_eps as f32;
        for i in 0..nr {
            kernel::rmsnorm(&self.hs[i * h..(i + 1) * h], &model.norm, eps, &mut self.hns[i * h..(i + 1) * h]);
        }
        if device {
            // The batched head is a host kernel. The sequential loop's head on
            // this tier is the device's single-row gemv, so the verify reads its
            // logits one row at a time through the same entry point - the
            // acceptance test's only chance to stay the same rule the sequential
            // loop applies.
            for i in 0..nr {
                model.logits(sh, &self.hns[i * h..(i + 1) * h], &mut self.lg[i * v..(i + 1) * v])?;
            }
        } else {
            model.logits_batch(sh, &self.hns[..nr * h], &mut self.lg[..nr * v])?;
        }

        // 3. Accept the longest prefix the trunk agrees with, then the token the
        //    last kept row sampled. That is `m` rows and `m` new tokens.
        let mut n = 0usize;
        while n < k && argmax(&self.lg[n * v..(n + 1) * v]) as u32 == self.draft[n] {
            n += 1;
        }
        let mut commit: Vec<u32> = self.draft[..n].to_vec();
        commit.push(argmax(&self.lg[n * v..(n + 1) * v]) as u32);
        if let Some(i) = commit.iter().position(|&t| stop(t)) {
            // Cut the stop token out, not in: the sequential loop breaks on it
            // without emitting it, so a speculating turn that ended one token
            // longer would not be the same answer. When it is the block's first
            // token `m` lands at 0 and the round keeps nothing.
            commit.truncate(i);
            self.stopped = true;
        }
        let m = commit.len();

        // 4. Roll back what the block fed on trust. The rows past `m` were
        //    computed against tokens that are not in the prefix, so the caches go
        //    back to before the verify and the kept rows run again - which is also
        //    what leaves `hprev` holding a real hidden for the next round. A round
        //    that keeps every row it fed needs neither, and at ~97% first-draft
        //    acceptance that is the common case.
        //
        //    The drafter keeps one row per consumed token that survived, which is
        //    `m`, except when every draft was right: the token the trunk corrected
        //    or added at the end has no drafter row, and rather than pay a head
        //    pass for a token it already knows the answer to the driver lets its
        //    own cache stay contiguous and skips that position. Acceptance, not
        //    correctness, is what that costs.
        rewind(&mut self.st, base + m.min(k));
        if m < nr {
            StateSave::restore_all(st, &self.slots);
            for (i, &t) in self.toks[..m].iter().enumerate() {
                let row = model.embed_row(sh, t as usize)?;
                self.hs[i * h..(i + 1) * h].copy_from_slice(&row);
            }
            if m > 0 {
                if device {
                    model.prefill_device(&mut self.hs[..m * h], st, pos)?;
                } else {
                    model.prefill(&mut self.hs[..m * h], st, pos);
                }
            }
        }
        if m > 0 {
            hprev.copy_from_slice(&self.hs[(m - 1) * h..m * h]);
        }
        ids.extend_from_slice(&commit);
        // The stage table divides its decode window by rows fed, and a round feeds
        // `m` of them in one batched pass that `Model::forward` never sees. Without
        // this the denominator sits at one per round and every stage reads as
        // milliseconds per round.
        for _ in 0..m {
            crate::prof::tick_decode_step();
        }
        self.rounds += 1;
        self.proposed += k;
        self.accepted += n;
        self.committed += m;
        Ok(m)
    }

    /// What the block bought, in the shape the telemetry line wants it: the
    /// draft acceptance and the tokens per round, which is the speedup.
    pub fn report(&self) -> String {
        if self.rounds == 0 {
            return String::new();
        }
        format!(
            "mtp: {} rounds, {}/{} drafts accepted, {:.2} tokens/round",
            self.rounds,
            self.accepted,
            self.proposed,
            self.committed as f64 / self.rounds as f64
        )
    }
}
