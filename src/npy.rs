//! Minimal .npy reader, for the golden fixtures only.

use crate::config::Result;

pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// Text following `key` in the header dict.
fn rest_after<'a>(head: &'a str, key: &str) -> &'a str {
    match head.find(key) {
        Some(i) => head[i + key.len()..].trim_start_matches([' ', ':']),
        None => "",
    }
}

/// `'key': 'value'` -> value.
fn quoted<'a>(head: &'a str, key: &str) -> Option<&'a str> {
    let rest = rest_after(head, key);
    if !rest.starts_with('\'') {
        return None;
    }
    let body = &rest[1..];
    let end = body.find('\'')?;
    Some(&body[..end])
}

/// `'key': (a, b, )` -> the numbers, in order.
fn ints(head: &str, key: &str) -> Vec<usize> {
    let rest = rest_after(head, key);
    let Some(open) = rest.find('(') else {
        return Vec::new();
    };
    let body = &rest[open + 1..];
    let closed = body.find(')').unwrap_or(body.len());
    body[..closed]
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect()
}

impl Npy {
    pub fn read(path: &str) -> Result<Npy> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if bytes.len() < 10 || &bytes[1..6] != b"NUMPY" {
            return Err(format!("{path}: not .npy"));
        }
        // .npy framing: 6 magic bytes, 2 version bytes, then the header length -
        // uint16 in v1.0 (what numpy writes) and uint32 from v2.0 on.
        let v = bytes[6];
        let (hlen, base) = match v {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
            2 | 3 => (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12usize,
            ),
            other => return Err(format!("{path}: npy version {other}")),
        };
        if bytes.len() < base + hlen {
            return Err(format!("{path}: truncated npy header"));
        }
        // numpy writes the header as a Python dict, not JSON: single quotes,
        // `False`, and tuples for shape. Reading those three fields by hand beats
        // teaching serde_json about all of them.
        let head = std::str::from_utf8(&bytes[base..base + hlen])
            .map_err(|_| "npy: header is not utf8".to_string())?;
        let descr = quoted(head, "'descr'").ok_or_else(|| format!("{path}: no descr in header"))?.to_string();
        let shape = ints(head, "'shape'");
        if rest_after(head, "'fortran_order'").starts_with('T') {
            return Err(format!("{path}: fortran-ordered arrays are not supported"));
        }
        if !descr.starts_with('<') {
            return Err(format!("{path}: big-endian descr `{descr}` is not supported"));
        }
        let payload = &bytes[base + hlen..];
        let n: usize = shape.iter().product::<usize>().max(1);
        let mut data = Vec::with_capacity(n);
        match descr.as_str() {
            "<f8" | "<i8" => {
                for c in 0..payload.len() / 8 {
                    let raw = u64::from_le_bytes(payload[c * 8..c * 8 + 8].try_into().unwrap());
                    data.push(if descr == "<f8" {
                        f64::from_bits(raw)
                    } else {
                        (raw as i64) as f64
                    });
                }
            }
            "<f4" => {
                for c in 0..payload.len() / 4 {
                    data.push(f32::from_bits(u32::from_le_bytes(payload[c * 4..c * 4 + 4].try_into().unwrap())) as f64);
                }
            }
            "<i4" | "<u4" => {
                for c in 0..payload.len() / 4 {
                    let raw = u32::from_le_bytes(payload[c * 4..c * 4 + 4].try_into().unwrap());
                    data.push(if descr == "<i4" {
                        (raw as i32) as f64
                    } else {
                        raw as f64
                    });
                }
            }
            "<i2" | "<u2" | "<i1" | "<u1" => {
                let w = if descr.ends_with('1') { 1 } else { 2 };
                for c in 0..payload.len() / w {
                    let raw = if w == 1 {
                        payload[c] as u64
                    } else {
                        u16::from_le_bytes([payload[c * 2], payload[c * 2 + 1]]) as u64
                    };
                    let signed = descr.starts_with("<i");
                    data.push(if signed {
                        match w {
                            1 => (raw as u8) as i8 as f64,
                            _ => (raw as u16) as i16 as f64,
                        }
                    } else {
                        raw as f64
                    });
                }
            }
            other => return Err(format!("{path}: unhandled npy dtype {other}")),
        }
        if shape.is_empty() {
            return Err(format!("{path}: empty shape"));
        }
        Ok(Npy { shape, data })
    }

    pub fn as2(&self) -> (usize, usize) {
        match self.shape.len() {
            1 => (self.shape[0], 1),
            n => (self.shape[n - 2], self.shape[n - 1]),
        }
    }

    /// Biggest absolute difference against a flat f32 candidate, plus the index.
    pub fn max_abs_diff(&self, got: &[f32]) -> (f64, usize) {
        let mut best = 0f64;
        let mut at = 0usize;
        for (i, &w) in self.data.iter().enumerate() {
            if i >= got.len() {
                break;
            }
            let d = (w - got[i] as f64).abs();
            if d > best {
                best = d;
                at = i;
            }
        }
        (best, at)
    }
}
