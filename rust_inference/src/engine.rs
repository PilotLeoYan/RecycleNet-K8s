//! ONNX Runtime inference engine abstraction and implementation.
//!
//! Follows SOLID principles:
//! - Dependency Inversion Principle: `InferenceBackend` trait decouples
//!   the HTTP layer from concrete ONNX Runtime specifics.
//! - Single Responsibility: discrete responsibilities for environment
//!   resolution, graph evaluation, and probability normalization.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ndarray::Array4;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::preprocess::{ImagePreprocessor, PreprocessError};

/// Default 6-class labels for TrashNet dataset.
pub const DEFAULT_CLASSES: [&str; 6] = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
];

/// Errors produced during inference engine lifecycle and execution.
#[derive(Error, Debug)]
pub enum EngineError {
    #[error("ONNX Runtime initialization error: {0}")]
    OrtInitError(String),
    #[error("ONNX Runtime session error: {0}")]
    OrtSessionError(String),
    #[error("Model file not found at: {0}")]
    ModelNotFound(PathBuf),
    #[error("Image preprocessing failed: {0}")]
    PreprocessError(#[from] PreprocessError),
    #[error("Failed to extract output tensor: {0}")]
    OutputExtractionError(String),
    #[error("Class mapping error: {0}")]
    ClassMappingError(String),
}

/// Structured inference result matching the Python API output contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PredictionResult {
    pub predicted_class: String,
    pub confidence: f32,
    pub class_probabilities: HashMap<String, f32>,
}

/// Abstract backend interface for waste classification inference.
pub trait InferenceBackend: Send + Sync {
    /// Returns true if the model is loaded and ready to serve inference.
    fn is_ready(&self) -> bool;

    /// Returns the list of recognized class names.
    fn classes(&self) -> &[String];

    /// Returns the path to the loaded model.
    fn model_path(&self) -> &Path;

    /// Runs forward inference on a normalized 4D tensor `(1, 3, 224, 224)`
    /// returning `(logits, softmax_probabilities)`.
    fn run_tensor(&self, tensor: &Array4<f32>) -> Result<(Vec<f32>, Vec<f32>), EngineError>;

    /// Decodes, preprocesses, and evaluates raw image bytes.
    fn predict(&self, image_bytes: &[u8]) -> Result<PredictionResult, EngineError>;
}

static ORT_ENV: OnceLock<Environment> = OnceLock::new();

/// Automatically locate ONNX Runtime dynamic library if not already configured.
fn resolve_ort_dylib() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("ORT_DYLIB_PATH") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }

    let candidates = [
        ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        "../.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        "../../.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        "/usr/lib/libonnxruntime.so",
        "/usr/local/lib/libonnxruntime.so",
        "/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
    ];

    for candidate in &candidates {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

/// Retrieves or creates the global ONNX Runtime environment.
fn get_or_init_ort_env() -> Result<&'static Environment, EngineError> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env);
    }

    let dylib = resolve_ort_dylib();
    let builder = match dylib {
        Some(path) => {
            tracing::info!("Initializing ONNX Runtime from dynamic library: {:?}", path);
            std::env::set_var("ORT_DYLIB_PATH", &path);
            ort::init_from(path.to_str().unwrap_or_default())
                .map_err(|e| EngineError::OrtInitError(e.to_string()))?
        }
        None => {
            tracing::info!("Initializing default ONNX Runtime environment");
            ort::init()
        }
    };

    let built_env = builder
        .with_name("RecycleNet-Rust")
        .build()
        .map_err(|e| EngineError::OrtInitError(e.to_string()))?;

    let _ = ORT_ENV.set(built_env);
    Ok(ORT_ENV.get().expect("Environment should be set"))
}

/// Resolve the model path from environment variable or standard local paths.
pub fn resolve_model_path(custom_path: Option<PathBuf>) -> Result<PathBuf, EngineError> {
    if let Some(p) = custom_path {
        if p.exists() {
            return Ok(p);
        }
        return Err(EngineError::ModelNotFound(p));
    }

    if let Ok(env_path) = std::env::var("RECYCLENET_MODEL_PATH") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Ok(p);
        }
    }

    let candidates = [
        "models/recyclenet.onnx",
        "../models/recyclenet.onnx",
        "../../models/recyclenet.onnx",
    ];

    for candidate in &candidates {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Ok(p);
        }
    }

    Err(EngineError::ModelNotFound(PathBuf::from("models/recyclenet.onnx")))
}

/// Load classes mapping from `classes_mapping.json` located next to model.
pub fn load_class_mapping(model_dir: &Path) -> Vec<String> {
    let mapping_path = model_dir.join("classes_mapping.json");
    if mapping_path.exists() {
        if let Ok(file) = File::open(&mapping_path) {
            let reader = BufReader::new(file);
            if let Ok(map) = serde_json::from_reader::<_, HashMap<String, String>>(reader) {
                let mut indexed_classes: Vec<(usize, String)> = map
                    .into_iter()
                    .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v)))
                    .collect();
                indexed_classes.sort_by_key(|(idx, _)| *idx);
                if !indexed_classes.is_empty() {
                    return indexed_classes.into_iter().map(|(_, name)| name).collect();
                }
            }
        }
    }

    DEFAULT_CLASSES.iter().map(|s| s.to_string()).collect()
}

/// Numerically stable softmax computation for 1D slices.
pub fn compute_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();

    if sum_exp == 0.0 || sum_exp.is_nan() {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }

    exps.into_iter().map(|v| v / sum_exp).collect()
}

/// Concrete ONNX Runtime inference engine.
pub struct OrtInferenceEngine {
    session: Mutex<Session>,
    input_name: String,
    classes: Vec<String>,
    model_path: PathBuf,
    preprocessor: ImagePreprocessor,
}

impl OrtInferenceEngine {
    /// Instantiate a new `OrtInferenceEngine` from a file path or standard lookup.
    pub fn new(custom_model_path: Option<PathBuf>) -> Result<Self, EngineError> {
        let model_path = resolve_model_path(custom_model_path)?;
        let env = get_or_init_ort_env()?;

        let session = Session::builder(env)
            .map_err(|e| EngineError::OrtSessionError(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| EngineError::OrtSessionError(e.to_string()))?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "image".to_string());

        let model_dir = model_path.parent().unwrap_or_else(|| Path::new("."));
        let classes = load_class_mapping(model_dir);

        tracing::info!(
            "Loaded model from {:?}, input_name='{}', classes={:?}",
            model_path,
            input_name,
            classes
        );

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            classes,
            model_path,
            preprocessor: ImagePreprocessor::new(),
        })
    }

    /// Provide a custom preprocessor if needed.
    pub fn with_preprocessor(mut self, preprocessor: ImagePreprocessor) -> Self {
        self.preprocessor = preprocessor;
        self
    }
}

impl InferenceBackend for OrtInferenceEngine {
    fn is_ready(&self) -> bool {
        true
    }

    fn classes(&self) -> &[String] {
        &self.classes
    }

    fn model_path(&self) -> &Path {
        &self.model_path
    }

    fn run_tensor(&self, tensor: &Array4<f32>) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
        let ort_tensor = Tensor::from_array(tensor.clone())
            .map_err(|e| EngineError::OrtSessionError(e.to_string()))?;

        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => ort_tensor])
            .map_err(|e| EngineError::OrtSessionError(e.to_string()))?;

        if outputs.len() == 0 {
            return Err(EngineError::OutputExtractionError(
                "Model produced 0 output tensors".to_string(),
            ));
        }

        let (_shape, logits_slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| EngineError::OutputExtractionError(e.to_string()))?;

        let logits: Vec<f32> = logits_slice.to_vec();
        let probs = compute_softmax(&logits);

        Ok((logits, probs))
    }

    fn predict(&self, image_bytes: &[u8]) -> Result<PredictionResult, EngineError> {
        let tensor = self.preprocessor.preprocess(image_bytes)?;
        let (_logits, probs) = self.run_tensor(&tensor)?;

        let (top1_idx, &top1_conf) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((0, &0.0));

        let predicted_class = self
            .classes
            .get(top1_idx)
            .cloned()
            .unwrap_or_else(|| format!("class_{top1_idx}"));

        let mut class_probabilities = HashMap::new();
        for (idx, &p) in probs.iter().enumerate() {
            let class_name = self
                .classes
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("class_{idx}"));
            let rounded_p = (p * 1_000_000.0).round() / 1_000_000.0;
            class_probabilities.insert(class_name, rounded_p);
        }

        let rounded_conf = (top1_conf * 1_000_000.0).round() / 1_000_000.0;

        Ok(PredictionResult {
            predicted_class,
            confidence: rounded_conf,
            class_probabilities,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_softmax_uniform() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let probs = compute_softmax(&logits);
        assert_eq!(probs.len(), 4);
        for p in &probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_softmax_numerical_stability() {
        // High logits that would overflow simple exp()
        let logits = vec![1000.0, 1001.0];
        let probs = compute_softmax(&logits);
        assert_eq!(probs.len(), 2);
        assert!(!probs[0].is_nan());
        assert!(!probs[1].is_nan());
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_default_classes() {
        assert_eq!(DEFAULT_CLASSES.len(), 6);
        assert_eq!(DEFAULT_CLASSES[0], "cardboard");
        assert_eq!(DEFAULT_CLASSES[5], "trash");
    }

    #[test]
    fn test_engine_initialization_if_model_exists() {
        let model_path = resolve_model_path(None);
        if let Ok(path) = model_path {
            let engine = OrtInferenceEngine::new(Some(path)).unwrap();
            assert!(engine.is_ready());
            assert_eq!(engine.classes().len(), 6);

            let dummy_tensor = Array4::<f32>::zeros((1, 3, 224, 224));
            let (logits, probs) = engine.run_tensor(&dummy_tensor).unwrap();
            assert_eq!(logits.len(), 6);
            assert_eq!(probs.len(), 6);
            let sum: f32 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}
