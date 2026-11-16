//! Image preprocessing plugin: resize and convert images before vision processing.
//!
//! Implements the "Efficient Browser Use" mod: shrinks screenshots to reduce
//! token consumption. Default resize factor is 0.5x (75% token reduction).
//!
//! Supports PNG, JPEG, and WebP formats via the `image` crate.

use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use image::{ImageFormat, ImageReader};
use std::io::Cursor;
use std::sync::{Arc, RwLock};

/// Configuration for image preprocessing.
#[derive(Clone, Debug)]
pub struct ImagePreprocessConfig {
    /// Resize factor (0.5 = half dimensions, 75% fewer pixels).
    pub resize_factor: f32,
    /// Output format (defaults to PNG).
    pub output_format: ImageOutputFormat,
    /// Maximum dimension (width or height). Images larger than this are scaled down.
    pub max_dimension: u32,
    /// JPEG/WebP quality (1-100). Only applies to lossy formats.
    pub quality: u8,
}

impl Default for ImagePreprocessConfig {
    fn default() -> Self {
        Self {
            resize_factor: 0.5,
            output_format: ImageOutputFormat::Png,
            max_dimension: 2048,
            quality: 85,
        }
    }
}

/// Output image format.
#[derive(Clone, Debug, PartialEq)]
pub enum ImageOutputFormat {
    Png,
    Jpeg,
    WebP,
}

/// Image preprocessing service.
pub struct ImagePreprocessService {
    config: RwLock<ImagePreprocessConfig>,
}

impl ImagePreprocessService {
    pub fn new() -> Self {
        Self {
            config: RwLock::new(ImagePreprocessConfig::default()),
        }
    }

    pub fn config(&self) -> ImagePreprocessConfig {
        self.config.read().unwrap().clone()
    }

    pub fn set_config(&self, config: ImagePreprocessConfig) {
        *self.config.write().unwrap() = config;
    }

    /// Resize an image from raw bytes. Returns the processed image bytes.
    pub fn process(&self, input: &[u8]) -> Result<Vec<u8>, ImagePreprocessError> {
        let config = self.config();

        let reader = ImageReader::new(Cursor::new(input))
            .with_guessed_format()
            .map_err(|e| ImagePreprocessError::Decode(e.to_string()))?;

        let _format = reader.format().unwrap_or(ImageFormat::Png);
        let mut img = reader
            .decode()
            .map_err(|e| ImagePreprocessError::Decode(e.to_string()))?;

        let (w, h) = (img.width(), img.height());

        let mut target_w = (w as f32 * config.resize_factor) as u32;
        let mut target_h = (h as f32 * config.resize_factor) as u32;

        if target_w > config.max_dimension {
            let scale = config.max_dimension as f32 / target_w as f32;
            target_w = config.max_dimension;
            target_h = (target_h as f32 * scale) as u32;
        }
        if target_h > config.max_dimension {
            let scale = config.max_dimension as f32 / target_h as f32;
            target_h = config.max_dimension;
            target_w = (target_w as f32 * scale) as u32;
        }

        target_w = target_w.max(1);
        target_h = target_h.max(1);

        if target_w != w || target_h != h {
            img = img.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);
        }

        let mut output = Cursor::new(Vec::new());
        match config.output_format {
            ImageOutputFormat::Png => {
                img.write_to(&mut output, ImageFormat::Png)
                    .map_err(|e| ImagePreprocessError::Encode(e.to_string()))?;
            }
            ImageOutputFormat::Jpeg => {
                let rgb = img.to_rgb8();
                rgb.write_to(&mut output, ImageFormat::Jpeg)
                    .map_err(|e| ImagePreprocessError::Encode(e.to_string()))?;
            }
            ImageOutputFormat::WebP => {
                img.write_to(&mut output, ImageFormat::WebP)
                    .map_err(|e| ImagePreprocessError::Encode(e.to_string()))?;
            }
        }

        Ok(output.into_inner())
    }

    /// Get the estimated token savings from resizing.
    /// Rough heuristic: tokens scale with pixel count.
    pub fn estimated_savings(&self, original_w: u32, original_h: u32) -> f32 {
        let config = self.config();
        let original_pixels = original_w as f32 * original_h as f32;
        let mut new_w = (original_w as f32 * config.resize_factor) as u32;
        let mut new_h = (original_h as f32 * config.resize_factor) as u32;
        if new_w > config.max_dimension {
            let scale = config.max_dimension as f32 / new_w as f32;
            new_w = config.max_dimension;
            new_h = (new_h as f32 * scale) as u32;
        }
        if new_h > config.max_dimension {
            let scale = config.max_dimension as f32 / new_h as f32;
            new_h = config.max_dimension;
            new_w = (new_w as f32 * scale) as u32;
        }
        let new_pixels = new_w.max(1) as f32 * new_h.max(1) as f32;
        1.0 - (new_pixels / original_pixels)
    }
}

impl Default for ImagePreprocessService {
    fn default() -> Self {
        Self::new()
    }
}

impl Service for ImagePreprocessService {
    fn name(&self) -> &'static str {
        "image-preprocess"
    }
}

/// Errors from image preprocessing.
#[derive(Debug)]
pub enum ImagePreprocessError {
    Decode(String),
    Encode(String),
    InvalidConfig(String),
}

impl std::fmt::Display for ImagePreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "image decode error: {}", e),
            Self::Encode(e) => write!(f, "image encode error: {}", e),
            Self::InvalidConfig(e) => write!(f, "invalid config: {}", e),
        }
    }
}

impl std::error::Error for ImagePreprocessError {}

/// Image preprocessing plugin.
pub struct ImagePreprocessPlugin {
    config: ImagePreprocessConfig,
}

impl ImagePreprocessPlugin {
    pub fn new(config: ImagePreprocessConfig) -> Self {
        Self { config }
    }
}

impl Default for ImagePreprocessPlugin {
    fn default() -> Self {
        Self::new(ImagePreprocessConfig::default())
    }
}

impl Plugin for ImagePreprocessPlugin {
    fn name(&self) -> &'static str {
        "image-preprocess"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
        let service = ImagePreprocessService::new();
        service.set_config(self.config.clone());
        let svc = Arc::new(service);
        ctx.register_service("image-preprocess", ImagePreprocessServiceHandle(svc.clone()))?;

        ctx.on_intercept("image/process", move |event| {
            if let Some(input) = event.payload::<Vec<u8>>() {
                match svc.process(input) {
                    Ok(output) => event.with_payload(output),
                    Err(_) => event,
                }
            } else {
                event
            }
        })?;

        ctx.effect(|| {
            // cleanup on unload
        })?;

        Ok(())
    }

    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> {
        ctx.emit(Event::new("image-preprocess/unloading", ()));
        Ok(())
    }
}

/// Wrapper to implement Service for Arc<ImagePreprocessService>.
struct ImagePreprocessServiceHandle(Arc<ImagePreprocessService>);

impl ImagePreprocessServiceHandle {
    #[allow(dead_code)]
    fn inner(&self) -> &ImagePreprocessService {
        &self.0
    }
}

impl Service for ImagePreprocessServiceHandle {
    fn name(&self) -> &'static str {
        "image-preprocess"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    fn make_test_png(w: u32, h: u32) -> Vec<u8> {
        let img = DynamicImage::new_rgb8(w, h);
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn test_resize_halves_dimensions() {
        let svc = ImagePreprocessService::new();
        let input = make_test_png(1000, 800);
        let output = svc.process(&input).unwrap();

        let reader = ImageReader::new(Cursor::new(&output))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(reader.width(), 500);
        assert_eq!(reader.height(), 400);
    }

    #[test]
    fn test_max_dimension_cap() {
        let svc = ImagePreprocessService::new();
        svc.set_config(ImagePreprocessConfig {
            resize_factor: 1.0,
            max_dimension: 512,
            ..Default::default()
        });
        let input = make_test_png(2000, 1000);
        let output = svc.process(&input).unwrap();

        let reader = ImageReader::new(Cursor::new(&output))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        assert!(reader.width() <= 512);
        assert!(reader.height() <= 512);
    }

    #[test]
    fn test_estimated_savings() {
        let svc = ImagePreprocessService::new();
        let savings = svc.estimated_savings(1000, 1000);
        assert!((savings - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_service_name() {
        let svc = ImagePreprocessService::new();
        assert_eq!(svc.name(), "image-preprocess");
    }
}
