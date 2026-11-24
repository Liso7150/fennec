use crate::config::VisionConfig;
use crate::model::{dense_half, Dense};
use crate::safetensors::Shards;
use image::DynamicImage;

struct VisionBlock {
    norm1: (Vec<f32>, Vec<f32>),
    qkv: Dense,
    qkv_bias: Vec<f32>,
    proj: Dense,
    proj_bias: Vec<f32>,
    norm2: (Vec<f32>, Vec<f32>),
    fc1: Dense,
    fc1_bias: Vec<f32>,
    fc2: Dense,
    fc2_bias: Vec<f32>,
}

pub struct VisionModel {
    cfg: VisionConfig,
    patch: Dense,
    patch_bias: Vec<f32>,
    pos: Vec<f32>,
    blocks: Vec<VisionBlock>,
    merger_norm: (Vec<f32>, Vec<f32>),
    merger_fc1: Dense,
    merger_fc1_bias: Vec<f32>,
    merger_fc2: Dense,
    merger_fc2_bias: Vec<f32>,
}

fn bias(sh: &Shards, name: &str) -> crate::config::Result<Vec<f32>> {
    sh.tensor_f32(name)
}

fn load_projection(sh: &Shards, name: &str, rows: usize, cols: usize) -> crate::config::Result<Dense> {
    let (d, got_rows, got_cols) = dense_half(sh, name)?;
    if got_rows != rows || got_cols != cols {
        return Err(format!("{name}: expected {rows}x{cols}, got {got_rows}x{got_cols}"));
    }
    Ok(d)
}

fn load_projection_flat(sh: &Shards, name: &str, rows: usize, cols: usize) -> crate::config::Result<Dense> {
    let info = sh.info(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if info.elems() != rows * cols {
        return Err(format!("{name}: expected {rows}x{cols} elements, got {}", info.elems()));
    }
    Ok(Dense::load(sh, name)?)
}

fn load_norm(sh: &Shards, prefix: &str) -> crate::config::Result<(Vec<f32>, Vec<f32>)> {
    Ok((bias(sh, &format!("{prefix}.weight"))?, bias(sh, &format!("{prefix}.bias"))?))
}

impl VisionModel {
    pub fn load(sh: &Shards, cfg: VisionConfig) -> crate::config::Result<Option<Self>> {
        let root = "model.visual";
        if !sh.contains(&format!("{root}.patch_embed.proj.weight")) {
            return Ok(None);
        }
        let c = cfg.in_channels;
        let p = cfg.patch_size;
        let patch_cols = c * cfg.temporal_patch_size * p * p;
        let patch = load_projection_flat(sh, &format!("{root}.patch_embed.proj.weight"), cfg.hidden_size, patch_cols)?;
        let patch_bias = bias(sh, &format!("{root}.patch_embed.proj.bias"))?;
        let pos = sh.tensor_f32(&format!("{root}.pos_embed.weight"))?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let b = format!("{root}.blocks.{i}");
            let qkv_rows = cfg.hidden_size * 3;
            let inter = cfg.intermediate_size;
            blocks.push(VisionBlock {
                norm1: load_norm(sh, &format!("{b}.norm1"))?,
                qkv: load_projection(sh, &format!("{b}.attn.qkv.weight"), qkv_rows, cfg.hidden_size)?,
                qkv_bias: bias(sh, &format!("{b}.attn.qkv.bias"))?,
                proj: load_projection(sh, &format!("{b}.attn.proj.weight"), cfg.hidden_size, cfg.hidden_size)?,
                proj_bias: bias(sh, &format!("{b}.attn.proj.bias"))?,
                norm2: load_norm(sh, &format!("{b}.norm2"))?,
                fc1: load_projection(sh, &format!("{b}.mlp.linear_fc1.weight"), inter, cfg.hidden_size)?,
                fc1_bias: bias(sh, &format!("{b}.mlp.linear_fc1.bias"))?,
                fc2: load_projection(sh, &format!("{b}.mlp.linear_fc2.weight"), cfg.hidden_size, inter)?,
                fc2_bias: bias(sh, &format!("{b}.mlp.linear_fc2.bias"))?,
            });
        }
        let merged = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        Ok(Some(Self {
            cfg: cfg.clone(),
            patch,
            patch_bias,
            pos,
            blocks,
            merger_norm: load_norm(sh, &format!("{root}.merger.norm"))?,
            merger_fc1: load_projection(sh, &format!("{root}.merger.linear_fc1.weight"), merged, merged)?,
            merger_fc1_bias: bias(sh, &format!("{root}.merger.linear_fc1.bias"))?,
            merger_fc2: load_projection(sh, &format!("{root}.merger.linear_fc2.weight"), cfg.out_hidden_size, merged)?,
            merger_fc2_bias: bias(sh, &format!("{root}.merger.linear_fc2.bias"))?,
        }))
    }

    pub fn output_tokens(&self, image: &[u8]) -> crate::config::Result<Vec<Vec<f32>>> {
        let img = image::load_from_memory(image).map_err(|e| format!("vision image decode: {e}"))?;
        let (w, h) = resized_dimensions(&img, self.cfg.patch_size * self.cfg.spatial_merge_size);
        let img = img.resize_exact(w, h, image::imageops::FilterType::Lanczos3).to_rgb32f();
        let gw = w as usize / self.cfg.patch_size;
        let gh = h as usize / self.cfg.patch_size;
        let merge = self.cfg.spatial_merge_size;
        let channels = self.cfg.in_channels;
        let ps = self.cfg.patch_size;
        let temporal = self.cfg.temporal_patch_size;
        let mut xs = Vec::with_capacity(gh * gw);
        let mut positions = Vec::with_capacity(gh * gw);
        for br in 0..gh / merge {
            for bc in 0..gw / merge {
                for ir in 0..merge {
                    for ic in 0..merge {
                        let r = br * merge + ir;
                        let c = bc * merge + ic;
                        let mut patch = vec![0f32; channels * temporal * ps * ps];
                        for ch in 0..channels {
                            for t in 0..temporal {
                                for y in 0..ps {
                                    for x in 0..ps {
                                        let px = img.get_pixel((c * ps + x) as u32, (r * ps + y) as u32);
                                        patch[((ch * temporal + t) * ps + y) * ps + x] =
                                            (px[ch] - 0.5) / 0.5;
                                    }
                                }
                            }
                        }
                        xs.push(patch);
                        positions.push((r, c));
                    }
                }
            }
        }
        let mut hidden = vec![0f32; xs.len() * self.cfg.hidden_size];
        for (i, x) in xs.iter().enumerate() {
            self.patch.gemv(self.cfg.hidden_size, x.len(), x, &mut hidden[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size], 1);
            for j in 0..self.cfg.hidden_size {
                hidden[i * self.cfg.hidden_size + j] += self.patch_bias[j];
            }
            let pos = interpolated_position(&self.pos, self.cfg.num_position_embeddings, self.cfg.hidden_size, gh, gw, positions[i].0, positions[i].1);
            for j in 0..self.cfg.hidden_size {
                hidden[i * self.cfg.hidden_size + j] += pos[j];
            }
        }
        let mut q = vec![0f32; xs.len() * self.cfg.hidden_size];
        let mut k = q.clone();
        let mut v = q.clone();
        for block in &self.blocks {
            let input = hidden.clone();
            let mut normed = vec![0f32; input.len()];
            for i in 0..xs.len() {
                layer_norm(&input[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size], &block.norm1.0, &block.norm1.1, &mut normed[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size]);
            }
            for i in 0..xs.len() {
                let row = &normed[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size];
                let mut fused = vec![0f32; self.cfg.hidden_size * 3];
                block.qkv.gemv(fused.len(), row.len(), row, &mut fused, 1);
                for j in 0..fused.len() { fused[j] += block.qkv_bias[j]; }
                q[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size].copy_from_slice(&fused[..self.cfg.hidden_size]);
                k[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size].copy_from_slice(&fused[self.cfg.hidden_size..self.cfg.hidden_size * 2]);
                v[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size].copy_from_slice(&fused[self.cfg.hidden_size * 2..]);
                apply_rope(&mut q[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size], positions[i], self.cfg.num_heads);
                apply_rope(&mut k[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size], positions[i], self.cfg.num_heads);
            }
            let mut attended = vec![0f32; input.len()];
            let hd = self.cfg.hidden_size / self.cfg.num_heads;
            for head in 0..self.cfg.num_heads {
                for i in 0..xs.len() {
                    let qi = &q[i * self.cfg.hidden_size + head * hd..i * self.cfg.hidden_size + (head + 1) * hd];
                    let mut scores = vec![0f32; xs.len()];
                    let mut max = f32::NEG_INFINITY;
                    for j in 0..xs.len() {
                        let kj = &k[j * self.cfg.hidden_size + head * hd..j * self.cfg.hidden_size + (head + 1) * hd];
                        scores[j] = dot(qi, kj) / (hd as f32).sqrt();
                        max = max.max(scores[j]);
                    }
                    let mut sum = 0.0;
                    for score in &mut scores { *score = (*score - max).exp(); sum += *score; }
                    for j in 0..xs.len() {
                        let weight = scores[j] / sum;
                        let vj = &v[j * self.cfg.hidden_size + head * hd..j * self.cfg.hidden_size + (head + 1) * hd];
                        for d in 0..hd { attended[i * self.cfg.hidden_size + head * hd + d] += weight * vj[d]; }
                    }
                }
            }
            let mut projected = vec![0f32; input.len()];
            for i in 0..xs.len() {
                let row = &attended[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size];
                block.proj.gemv(self.cfg.hidden_size, row.len(), row, &mut projected[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size], 1);
                for j in 0..self.cfg.hidden_size { projected[i * self.cfg.hidden_size + j] += block.proj_bias[j]; }
            }
            for i in 0..hidden.len() { hidden[i] = input[i] + projected[i]; }
            let input = hidden.clone();
            for i in 0..xs.len() {
                layer_norm(&input[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size], &block.norm2.0, &block.norm2.1, &mut normed[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size]);
                let row = &normed[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size];
                let mut mid = vec![0f32; self.cfg.intermediate_size];
                block.fc1.gemv(mid.len(), row.len(), row, &mut mid, 1);
                for j in 0..mid.len() { mid[j] = gelu_tanh(mid[j] + block.fc1_bias[j]); }
                let out = &mut projected[i * self.cfg.hidden_size..(i + 1) * self.cfg.hidden_size];
                block.fc2.gemv(self.cfg.hidden_size, mid.len(), &mid, out, 1);
                for j in 0..self.cfg.hidden_size { hidden[i * self.cfg.hidden_size + j] = input[i * self.cfg.hidden_size + j] + out[j] + block.fc2_bias[j]; }
            }
        }
        let merged = merge * merge;
        let mut output = Vec::with_capacity(xs.len() / merged);
        for i in 0..xs.len() / merged {
            let mut row = vec![0f32; self.cfg.hidden_size * merged];
            for j in 0..merged { row[j * self.cfg.hidden_size..(j + 1) * self.cfg.hidden_size].copy_from_slice(&hidden[(i * merged + j) * self.cfg.hidden_size..(i * merged + j + 1) * self.cfg.hidden_size]); }
            let mut norm = vec![0f32; row.len()];
            for j in 0..merged {
                layer_norm(
                    &row[j * self.cfg.hidden_size..(j + 1) * self.cfg.hidden_size],
                    &self.merger_norm.0,
                    &self.merger_norm.1,
                    &mut norm[j * self.cfg.hidden_size..(j + 1) * self.cfg.hidden_size],
                );
            }
            let mut mid = vec![0f32; row.len()];
            self.merger_fc1.gemv(row.len(), row.len(), &norm, &mut mid, 1);
            for j in 0..mid.len() { mid[j] = gelu_tanh(mid[j] + self.merger_fc1_bias[j]); }
            let mut out = vec![0f32; self.cfg.out_hidden_size];
            self.merger_fc2.gemv(self.cfg.out_hidden_size, mid.len(), &mid, &mut out, 1);
            for j in 0..out.len() { out[j] += self.merger_fc2_bias[j]; }
            output.push(out);
        }
        Ok(output)
    }
}

fn resized_dimensions(img: &DynamicImage, multiple: usize) -> (u32, u32) {
    let w = img.width().max(1) as usize;
    let h = img.height().max(1) as usize;
    let scale = (multiple as f32 / w.min(h) as f32).max(1.0);
    let mut nw = (w as f32 * scale).round() as usize;
    let mut nh = (h as f32 * scale).round() as usize;
    nw = nw.div_ceil(multiple) * multiple;
    nh = nh.div_ceil(multiple) * multiple;
    (nw as u32, nh as u32)
}

fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32]) {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var = x.iter().map(|v| { let d = *v - mean; d * d }).sum::<f32>() / x.len() as f32;
    let inv = (var + 1e-6).sqrt().recip();
    for i in 0..x.len() { out[i] = (x[i] - mean) * inv * weight[i] + bias[i]; }
}

fn interpolated_position(pos: &[f32], npos: usize, dim: usize, gh: usize, gw: usize, r: usize, c: usize) -> Vec<f32> {
    let side = (npos as f32).sqrt() as usize;
    let sy = r as f32 * (side.saturating_sub(1)) as f32 / (gh.saturating_sub(1).max(1)) as f32;
    let sx = c as f32 * (side.saturating_sub(1)) as f32 / (gw.saturating_sub(1).max(1)) as f32;
    let y0 = sy.floor() as usize; let x0 = sx.floor() as usize;
    let y1 = (y0 + 1).min(side - 1); let x1 = (x0 + 1).min(side - 1);
    let fy = sy - y0 as f32; let fx = sx - x0 as f32;
    (0..dim).map(|j| {
        pos[y0 * side * dim + x0 * dim + j] * (1.0 - fy) * (1.0 - fx)
            + pos[y1 * side * dim + x0 * dim + j] * fy * (1.0 - fx)
            + pos[y0 * side * dim + x1 * dim + j] * (1.0 - fy) * fx
            + pos[y1 * side * dim + x1 * dim + j] * fy * fx
    }).collect()
}

fn apply_rope(x: &mut [f32], (r, c): (usize, usize), heads: usize) {
    let hd = x.len() / heads;
    let half = hd / 2;
    let quarter = half / 2;
    let theta = 10000.0f32;
    for h in 0..heads {
        let row = &mut x[h * hd..(h + 1) * hd];
        for i in 0..quarter {
            let inv = theta.powf(-2.0 * i as f32 / half as f32);
            let a = r as f32 * inv; let b = c as f32 * inv;
            let (ca, sa, cb, sb) = (a.cos(), a.sin(), b.cos(), b.sin());
            let (ar, ai, br, bi) = (row[i], row[i + quarter], row[i + half], row[i + half + quarter]);
            row[i] = ar * ca - ai * sa;
            row[i + quarter] = ar * sa + ai * ca;
            row[i + half] = br * cb - bi * sb;
            row[i + half + quarter] = br * sb + bi * cb;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}
