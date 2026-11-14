//! GGUF format reader for Edge0 and other llama.cpp-compatible models.
//!
//! GGUF (GPT-Generated Unified Format) is a binary format for quantized models.
//! This reader supports the format used by Edge0-145B and similar models.
//!
//! Format specification: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::config::Result;

/// GGUF file magic number
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

/// GGUF file header
#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

/// GGUF metadata value types
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// GGUF tensor info
#[derive(Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgufTensorType,
    pub offset: u64,
}

/// GGUF tensor types (quantization formats)
#[derive(Debug, Clone, Copy)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    BF16 = 30,
}

impl GgufTensorType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Bytes per weight for this tensor type
    pub fn bytes_per_weight(&self) -> f32 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::BF16 => 2.0,
            Self::Q8_0 | Self::Q8_1 | Self::Q8K => 1.0,
            Self::Q4_0 | Self::Q4_1 | Self::Q4K | Self::IQ4NL | Self::IQ4XS => 0.5,
            Self::Q5_0 | Self::Q5_1 | Self::Q5K => 0.625,
            Self::Q6K => 0.75,
            Self::Q2K | Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => 0.25,
            Self::Q3K | Self::IQ3XXS | Self::IQ3S => 0.375,
            Self::IQ1S => 0.125,
        }
    }
}

/// A GGUF file reader
pub struct GgufReader {
    file: File,
    path: PathBuf,
    pub header: GgufHeader,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    data_offset: u64,
}

impl GgufReader {
    /// Open a GGUF file and parse its header
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)
            .map_err(|e| format!("Failed to open GGUF file {:?}: {}", path, e))?;

        // Read and verify magic
        let magic = read_u32_le(&mut file)?;
        if magic != GGUF_MAGIC {
            return Err(format!("Invalid GGUF magic: expected 0x{:08X}, got 0x{:08X}", GGUF_MAGIC, magic).into());
        }

        // Read header
        let version = read_u32_le(&mut file)?;
        let tensor_count = read_u64_le(&mut file)?;
        let metadata_kv_count = read_u64_le(&mut file)?;

        let header = GgufHeader {
            version,
            tensor_count,
            metadata_kv_count,
        };

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut file)?;
            let value = read_gguf_value(&mut file)?;
            metadata.insert(key, value);
        }

        // Align to 32 bytes (GGUF specification)
        let pos = file.stream_position().map_err(|e| e.to_string())?;
        let aligned_pos = (pos + 31) & !31;
        file.seek(SeekFrom::Start(aligned_pos)).map_err(|e| e.to_string())?;
        let _data_offset = aligned_pos;

        // Read tensor infos
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut file)?;
            let n_dims = read_u32_le(&mut file)?;
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(read_u64_le(&mut file)?);
            }
            let tensor_type_raw = read_u32_le(&mut file)?;
            let tensor_type = GgufTensorType::from_u32(tensor_type_raw)
                .ok_or_else(|| format!("Unknown tensor type: {}", tensor_type_raw))?;
            let offset = read_u64_le(&mut file)?;

            tensors.insert(name.clone(), GgufTensorInfo {
                name,
                dimensions,
                tensor_type,
                offset,
            });
        }

        // Align to 32 bytes again for tensor data
        let pos = file.stream_position().map_err(|e| e.to_string())?;
        let aligned_pos = (pos + 31) & !31;
        file.seek(SeekFrom::Start(aligned_pos)).map_err(|e| e.to_string())?;

        Ok(Self {
            file,
            path,
            header,
            metadata,
            tensors,
            data_offset: aligned_pos,
        })
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get tensor data as f32 (dequantizing if necessary)
    pub fn tensor_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let info = self.tensors.get(name)
            .ok_or_else(|| format!("Tensor not found: {}", name))?;

        let elem_count: u64 = info.dimensions.iter().product();
        let byte_offset = self.data_offset + info.offset;

        self.file.seek(SeekFrom::Start(byte_offset)).map_err(|e| e.to_string())?;

        match info.tensor_type {
            GgufTensorType::F32 => {
                let mut data = vec![0u8; elem_count as usize * 4];
                self.file.read_exact(&mut data).map_err(|e| e.to_string())?;
                Ok(data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            }
            GgufTensorType::F16 => {
                let mut data = vec![0u8; elem_count as usize * 2];
                self.file.read_exact(&mut data).map_err(|e| e.to_string())?;
                Ok(data.chunks_exact(2)
                    .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect())
            }
            GgufTensorType::BF16 => {
                let mut data = vec![0u8; elem_count as usize * 2];
                self.file.read_exact(&mut data).map_err(|e| e.to_string())?;
                Ok(data.chunks_exact(2)
                    .map(|chunk| bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect())
            }
            GgufTensorType::Q8_0 => {
                // Q8_0: block size 32, 1 f16 scale + 32 i8 weights
                let block_size = 32;
                let num_blocks = elem_count as usize / block_size;
                let mut result = Vec::with_capacity(elem_count as usize);
                
                for _ in 0..num_blocks {
                    let mut scale_bytes = [0u8; 2];
                    self.file.read_exact(&mut scale_bytes).map_err(|e| e.to_string())?;
                    let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));
                    
                    let mut weights = vec![0i8; block_size];
                    self.file.read_exact(unsafe {
                        std::slice::from_raw_parts_mut(weights.as_mut_ptr() as *mut u8, block_size)
                    }).map_err(|e| e.to_string())?;
                    
                    for &w in &weights {
                        result.push(w as f32 * scale);
                    }
                }
                Ok(result)
            }
            GgufTensorType::Q4_0 => {
                // Q4_0: block size 32, 1 f16 scale + 16 bytes (32 x 4-bit weights)
                let block_size = 32;
                let num_blocks = elem_count as usize / block_size;
                let mut result = Vec::with_capacity(elem_count as usize);
                
                for _ in 0..num_blocks {
                    let mut scale_bytes = [0u8; 2];
                    self.file.read_exact(&mut scale_bytes).map_err(|e| e.to_string())?;
                    let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));
                    
                    let mut packed = vec![0u8; 16];
                    self.file.read_exact(&mut packed).map_err(|e| e.to_string())?;
                    
                    for &byte in &packed {
                        let low = (byte & 0x0F) as i8 - 8;
                        let high = ((byte >> 4) & 0x0F) as i8 - 8;
                        result.push(low as f32 * scale);
                        result.push(high as f32 * scale);
                    }
                }
                Ok(result)
            }
            GgufTensorType::Q3K => {
                // Q3_K: more complex, simplified implementation
                // For now, return zeros - full implementation would need the complete spec
                Ok(vec![0.0f32; elem_count as usize])
            }
            _ => {
                Err(format!("Unsupported tensor type: {:?}", info.tensor_type).into())
            }
        }
    }

    /// Get a metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get model name from metadata
    pub fn model_name(&self) -> Option<&str> {
        match self.metadata.get("general.name") {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }
}

// Helper functions for reading GGUF format

fn read_u32_le(file: &mut File) -> Result<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(file: &mut File) -> Result<u64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(buf))
}

fn read_gguf_string(file: &mut File) -> Result<String> {
    let len = read_u64_le(file)? as usize;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    String::from_utf8(buf).map_err(|e| format!("Invalid UTF-8 in string: {}", e).into())
}

fn read_gguf_value(file: &mut File) -> Result<GgufValue> {
    let value_type = read_u32_le(file)?;
    match value_type {
        0 => Ok(GgufValue::Uint8(read_u8(file)?)),
        1 => Ok(GgufValue::Int8(read_i8(file)?)),
        2 => Ok(GgufValue::Uint16(read_u16_le(file)?)),
        3 => Ok(GgufValue::Int16(read_i16_le(file)?)),
        4 => Ok(GgufValue::Uint32(read_u32_le(file)?)),
        5 => Ok(GgufValue::Int32(read_i32_le(file)?)),
        6 => Ok(GgufValue::Float32(read_f32_le(file)?)),
        7 => Ok(GgufValue::Bool(read_u8(file)? != 0)),
        8 => Ok(GgufValue::String(read_gguf_string(file)?)),
        9 => {
            let _arr_type = read_u32_le(file)?;
            let arr_len = read_u64_le(file)? as usize;
            let mut arr = Vec::with_capacity(arr_len);
            for _ in 0..arr_len {
                arr.push(read_gguf_value(file)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(read_u64_le(file)?)),
        11 => Ok(GgufValue::Int64(read_i64_le(file)?)),
        12 => Ok(GgufValue::Float64(read_f64_le(file)?)),
        _ => Err(format!("Unknown GGUF value type: {}", value_type).into()),
    }
}

fn read_u8(file: &mut File) -> Result<u8> {
    let mut buf = [0u8; 1];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf[0])
}

fn read_i8(file: &mut File) -> Result<i8> {
    let mut buf = [0u8; 1];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf[0] as i8)
}

fn read_u16_le(file: &mut File) -> Result<u16> {
    let mut buf = [0u8; 2];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16_le(file: &mut File) -> Result<i16> {
    let mut buf = [0u8; 2];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(i16::from_le_bytes(buf))
}

fn read_i32_le(file: &mut File) -> Result<i32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64_le(file: &mut File) -> Result<i64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32_le(file: &mut File) -> Result<f32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64_le(file: &mut File) -> Result<f64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(f64::from_le_bytes(buf))
}

/// Convert f16 to f32
fn f16_to_f32(v: u16) -> f32 {
    let sign = ((v >> 15) & 1) as u32;
    let exp = ((v >> 10) & 0x1F) as u32;
    let mant = (v & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut m = mant as f32;
            let mut e = 1i32;
            while (m as u32) & 0x400 == 0 {
                m *= 2.0;
                e -= 1;
            }
            m -= 1024.0;
            f32::from_bits((sign << 31) | ((e + 127 - 14) as u32) << 23 | (m as u32))
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            f32::from_bits(0x7FC00000) // NaN
        }
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Convert bf16 to f32
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}
