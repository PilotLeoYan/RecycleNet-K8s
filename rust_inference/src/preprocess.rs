//! Image preprocessing module for RecycleNet Rust inference engine.
//!
//! Follows torchvision.transforms.v2 pipeline:
//!   ToImage -> ToDtype(float32, scale=True) -> Resize((224, 224)) -> Normalize(mean, std)
//! Ensures exact numerical parity with the Python training and FastAPI baseline.

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::Array4;
use thiserror::Error;

/// ImageNet normalization mean constants: [R, G, B].
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// ImageNet normalization standard deviation constants: [R, G, B].
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Default target spatial height for MobileNetV3.
pub const TARGET_HEIGHT: usize = 224;

/// Default target spatial width for MobileNetV3.
pub const TARGET_WIDTH: usize = 224;

/// Preprocessing error types.
#[derive(Error, Debug)]
pub enum PreprocessError {
    #[error("Failed to decode image from memory: {0}")]
    ImageDecodeError(String),
    #[error("Invalid image dimensions: {0}x{1}")]
    InvalidDimensions(u32, u32),
}

/// Image preprocessor providing transformation from raw image bytes
/// to normalized 4D tensors `(1, 3, 224, 224)` ready for ONNX Runtime.
#[derive(Debug, Clone, Copy)]
pub struct ImagePreprocessor {
    pub target_height: usize,
    pub target_width: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self {
            target_height: TARGET_HEIGHT,
            target_width: TARGET_WIDTH,
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
        }
    }
}

impl ImagePreprocessor {
    /// Create a new preprocessor with default ImageNet normalization and 224x224 target resolution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Preprocess raw image bytes (JPEG, PNG, etc.) into a normalized 4D tensor `(1, 3, 224, 224)`.
    pub fn preprocess(&self, image_bytes: &[u8]) -> Result<Array4<f32>, PreprocessError> {
        let dynamic_img = image::load_from_memory(image_bytes)
            .map_err(|e| PreprocessError::ImageDecodeError(e.to_string()))?;
        self.preprocess_image(&dynamic_img)
    }

    /// Preprocess an already decoded `DynamicImage` into a normalized 4D tensor `(1, 3, 224, 224)`.
    pub fn preprocess_image(
        &self,
        dynamic_img: &DynamicImage,
    ) -> Result<Array4<f32>, PreprocessError> {
        let (width, height) = dynamic_img.dimensions();
        if width == 0 || height == 0 {
            return Err(PreprocessError::InvalidDimensions(width, height));
        }

        // Avoid resampling when image is already target size for exact pixel-for-pixel parity
        let rgb_img = if width == self.target_width as u32
            && height == self.target_height as u32
        {
            dynamic_img.to_rgb8()
        } else {
            image::imageops::resize(
                &dynamic_img.to_rgb8(),
                self.target_width as u32,
                self.target_height as u32,
                FilterType::Triangle, // Bilinear / Triangle filter matching torchvision
            )
        };

        let raw_pixels = rgb_img.as_raw();
        let mut tensor = Array4::<f32>::zeros((1, 3, self.target_height, self.target_width));

        let mean_r = self.mean[0];
        let mean_g = self.mean[1];
        let mean_b = self.mean[2];
        let std_r = self.std[0];
        let std_g = self.std[1];
        let std_b = self.std[2];

        // Fill NCHW tensor: Channel 0 = Red, Channel 1 = Green, Channel 2 = Blue
        for y in 0..self.target_height {
            for x in 0..self.target_width {
                let pixel_idx = (y * self.target_width + x) * 3;
                let r = raw_pixels[pixel_idx] as f32 / 255.0;
                let g = raw_pixels[pixel_idx + 1] as f32 / 255.0;
                let b = raw_pixels[pixel_idx + 2] as f32 / 255.0;

                tensor[[0, 0, y, x]] = (r - mean_r) / std_r;
                tensor[[0, 1, y, x]] = (g - mean_g) / std_g;
                tensor[[0, 2, y, x]] = (b - mean_b) / std_b;
            }
        }

        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use std::io::Cursor;

    #[test]
    fn test_preprocessor_default_constants() {
        let preprocessor = ImagePreprocessor::new();
        assert_eq!(preprocessor.target_height, 224);
        assert_eq!(preprocessor.target_width, 224);
        assert_eq!(preprocessor.mean, [0.485, 0.456, 0.406]);
        assert_eq!(preprocessor.std, [0.229, 0.224, 0.225]);
    }

    #[test]
    fn test_preprocess_invalid_image_bytes() {
        let preprocessor = ImagePreprocessor::new();
        let invalid_bytes = b"not_an_image_file";
        let result = preprocessor.preprocess(invalid_bytes);
        assert!(result.is_err());
        match result {
            Err(PreprocessError::ImageDecodeError(_)) => {}
            _ => panic!("Expected ImageDecodeError"),
        }
    }

    #[test]
    fn test_preprocess_exact_size_black_image() {
        let preprocessor = ImagePreprocessor::new();
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(224, 224, Rgb([0, 0, 0]));

        let mut buffer = Vec::new();
        img.write_to(&mut Cursor::new(&mut buffer), image::ImageFormat::Png)
            .unwrap();

        let tensor = preprocessor.preprocess(&buffer).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);

        // Expected normalized values for black (0 / 255.0):
        // (0 - 0.485) / 0.229 = -2.1179039...
        // (0 - 0.456) / 0.224 = -2.0357143...
        // (0 - 0.406) / 0.225 = -1.8044444...
        let expected_r = -0.485 / 0.229;
        let expected_g = -0.456 / 0.224;
        let expected_b = -0.406 / 0.225;

        let diff_r = (tensor[[0, 0, 112, 112]] - expected_r).abs();
        let diff_g = (tensor[[0, 1, 112, 112]] - expected_g).abs();
        let diff_b = (tensor[[0, 2, 112, 112]] - expected_b).abs();

        assert!(diff_r < 1e-5, "R diff too large: {}", diff_r);
        assert!(diff_g < 1e-5, "G diff too large: {}", diff_g);
        assert!(diff_b < 1e-5, "B diff too large: {}", diff_b);
    }

    #[test]
    fn test_preprocess_resize_different_dimensions() {
        let preprocessor = ImagePreprocessor::new();
        // Create 100x150 white image
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(100, 150, Rgb([255, 255, 255]));

        let mut buffer = Vec::new();
        img.write_to(&mut Cursor::new(&mut buffer), image::ImageFormat::Png)
            .unwrap();

        let tensor = preprocessor.preprocess(&buffer).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);

        // Expected normalized values for white (255 / 255.0 = 1.0):
        // (1.0 - 0.485) / 0.229 = 2.248908...
        // (1.0 - 0.456) / 0.224 = 2.428571...
        // (1.0 - 0.406) / 0.225 = 2.640000...
        let expected_r = (1.0 - 0.485) / 0.229;
        let diff_r = (tensor[[0, 0, 50, 50]] - expected_r).abs();
        assert!(diff_r < 1e-4, "Resize R diff too large: {}", diff_r);
    }
}
