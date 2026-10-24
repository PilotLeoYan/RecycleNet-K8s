//! High-performance RecycleNet inference microservice in Rust with Axum.

//!

//! Provides identical REST API contracts to the Python FastAPI baseline:

//! - GET  /health   -> Health and model readiness status

//! - GET  /healthz  -> Kubernetes liveness probe

//! - GET  /ready    -> Kubernetes readiness probe

//! - GET  /info     -> Service metadata, architecture, classes, and git commit

//! - POST /predict  -> Multiclass waste inference (multipart, form-urlencoded, or JSON)

//! - POST /predict/ -> Alias for /predict



use std::collections::HashMap;

use std::net::SocketAddr;

use std::path::PathBuf;

use std::sync::Arc;



use axum::extract::{FromRequest, Multipart, Request, State};

use axum::http::{header, HeaderMap, StatusCode};

use axum::response::{IntoResponse, Json, Response};

use axum::routing::{get, post};

use axum::Router;

use base64::prelude::*;

use serde::{Deserialize, Serialize};

use serde_json::json;

use tower_http::cors::CorsLayer;



use rust_inference::engine::{InferenceBackend, OrtInferenceEngine};



/// Application shared state containing the inference backend abstraction.

#[derive(Clone)]

pub struct AppState {

    pub backend: Arc<dyn InferenceBackend>,

}



/// Telemetry metadata matching Python schema.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]

pub struct Telemetry {

    pub latitude: Option<f64>,

    pub longitude: Option<f64>,

}



/// Prediction response contract matching Python FastAPI `PredictionResponse`.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]

pub struct PredictionResponse {

    pub predicted_class: String,

    pub confidence: f32,

    pub class_probabilities: HashMap<String, f32>,

    pub telemetry: Option<Telemetry>,

}



/// Health response schema.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]

pub struct HealthResponse {

    pub status: String,

    pub model_ready: bool,

}



/// Readiness response schema.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]

pub struct ReadyResponse {

    pub status: String,

    pub model: String,

}



/// Metadata information response contract.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]

pub struct InfoResponse {

    pub service_name: String,

    pub version: String,

    pub git_commit: String,

    pub model_architecture: String,

    pub model_format: String,

    pub opset_version: u32,

    pub input_resolution: Vec<u32>,

    pub classes: Vec<String>,

    pub model_ready: bool,

}



/// Resolve Git commit hash from environment variable or git CLI.

pub fn get_git_commit() -> String {

    if let Ok(env_commit) = std::env::var("GIT_COMMIT") {

        let trimmed = env_commit.trim();

        if !trimmed.is_empty() {

            return trimmed.to_string();

        }

    }



    if let Ok(output) = std::process::Command::new("git")

        .args(["rev-parse", "--short", "HEAD"])

        .output()

    {

        if output.status.success() {

            if let Ok(s) = String::from_utf8(output.stdout) {

                let trimmed = s.trim();

                if !trimmed.is_empty() {

                    return trimmed.to_string();

                }

            }

        }

    }



    "unknown".to_string()

}



/// Kubernetes liveness probe handler.

pub async fn healthz_handler() -> impl IntoResponse {

    (StatusCode::OK, Json(json!({"status": "ok"})))

}



/// Kubernetes readiness probe handler.

pub async fn ready_handler(State(state): State<AppState>) -> impl IntoResponse {

    if state.backend.is_ready() {

        (

            StatusCode::OK,

            Json(json!({"status": "ready", "model": "loaded"})),

        )

    } else {

        (

            StatusCode::SERVICE_UNAVAILABLE,

            Json(json!({"status": "not_ready", "model": "loading"})),

        )

    }

}



/// General health check handler.

pub async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {

    let ready = state.backend.is_ready();

    let status_code = if ready {

        StatusCode::OK

    } else {

        StatusCode::SERVICE_UNAVAILABLE

    };

    let body = HealthResponse {

        status: if ready {

            "healthy".to_string()

        } else {

            "degraded".to_string()

        },

        model_ready: ready,

    };

    (status_code, Json(body))

}



/// Service and model metadata info handler.

pub async fn info_handler(State(state): State<AppState>) -> impl IntoResponse {

    let info = InfoResponse {

        service_name: "RecycleNet-API".to_string(),

        version: "0.1.0".to_string(),

        git_commit: get_git_commit(),

        model_architecture: "MobileNetV3".to_string(),

        model_format: "ONNX".to_string(),

        opset_version: 18,

        input_resolution: vec![224, 224],

        classes: state.backend.classes().to_vec(),

        model_ready: state.backend.is_ready(),

    };

    (StatusCode::OK, Json(info))

}



#[derive(Deserialize, Default)]

struct FormPayload {

    image_base64: Option<String>,

    latitude: Option<f64>,

    longitude: Option<f64>,

    telemetry: Option<String>,

}



#[derive(Deserialize, Default)]

struct JsonPayload {

    image_base64: Option<String>,

    latitude: Option<f64>,

    longitude: Option<f64>,

    telemetry: Option<serde_json::Value>,

}



/// Helper to parse telemetry coordinates from json string or explicit coordinates.

fn parse_telemetry(

    raw_telemetry: Option<&str>,

    latitude: Option<f64>,

    longitude: Option<f64>,

) -> Option<Telemetry> {

    if let Some(raw) = raw_telemetry {

        let trimmed = raw.trim();

        if !trimmed.is_empty() {

            if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {

                if let Some(obj) = val.as_object() {

                    let lat = obj.get("latitude").and_then(|v| v.as_f64());

                    let lon = obj.get("longitude").and_then(|v| v.as_f64());

                    if lat.is_some() || lon.is_some() {

                        return Some(Telemetry {

                            latitude: lat,

                            longitude: lon,

                        });

                    }

                }

            }

        }

    }



    if latitude.is_some() || longitude.is_some() {

        return Some(Telemetry {

            latitude,

            longitude,

        });

    }



    None

}



/// Multiclass inference prediction handler supporting multipart/form-data,

/// application/x-www-form-urlencoded, and application/json.

pub async fn predict_handler(

    State(state): State<AppState>,

    headers: HeaderMap,

    request: Request,

) -> Response {

    if !state.backend.is_ready() {

        return (

            StatusCode::SERVICE_UNAVAILABLE,

            Json(json!({"detail": "Model not loaded – service unavailable"})),

        )

            .into_response();

    }



    let content_type = headers

        .get(header::CONTENT_TYPE)

        .and_then(|v| v.to_str().ok())

        .unwrap_or("")

        .to_string();



    let mut image_bytes: Option<Vec<u8>> = None;

    let mut image_base64: Option<String> = None;

    let mut latitude: Option<f64> = None;

    let mut longitude: Option<f64> = None;

    let mut telemetry_str: Option<String> = None;



    if content_type.starts_with("multipart/form-data") {

        match Multipart::from_request(request, &state).await {

            Ok(mut multipart) => {

                while let Ok(Some(field)) = multipart.next_field().await {

                    let name = field.name().unwrap_or("").to_string();

                    match name.as_str() {

                        "file" => match field.bytes().await {

                            Ok(b) => {

                                if !b.is_empty() {

                                    image_bytes = Some(b.to_vec());

                                }

                            }

                            Err(e) => {

                                return (

                                    StatusCode::BAD_REQUEST,

                                    Json(json!({"detail": format!("Error reading file: {e}")})),

                                )

                                    .into_response();

                            }

                        },

                        "image_base64" => {

                            if let Ok(txt) = field.text().await {

                                let trimmed = txt.trim().to_string();

                                if !trimmed.is_empty() {

                                    image_base64 = Some(trimmed);

                                }

                            }

                        }

                        "latitude" => {

                            if let Ok(txt) = field.text().await {

                                latitude = txt.trim().parse::<f64>().ok();

                            }

                        }

                        "longitude" => {

                            if let Ok(txt) = field.text().await {

                                longitude = txt.trim().parse::<f64>().ok();

                            }

                        }

                        "telemetry" => {

                            if let Ok(txt) = field.text().await {

                                telemetry_str = Some(txt);

                            }

                        }

                        _ => {}

                    }

                }

            }

            Err(e) => {

                return (

                    StatusCode::BAD_REQUEST,

                    Json(json!({"detail": format!("Multipart error: {e}")})),

                )

                    .into_response();

            }

        }

    } else {

        let body_bytes = match axum::body::to_bytes(request.into_body(), 50 * 1024 * 1024).await {

            Ok(b) => b,

            Err(e) => {

                return (

                    StatusCode::BAD_REQUEST,

                    Json(json!({"detail": format!("Failed to read request body: {e}")})),

                )

                    .into_response();

            }

        };



        if content_type.starts_with("application/json") {

            if let Ok(payload) = serde_json::from_slice::<JsonPayload>(&body_bytes) {

                image_base64 = payload.image_base64;

                latitude = payload.latitude;

                longitude = payload.longitude;

                if let Some(tel) = payload.telemetry {

                    telemetry_str = Some(tel.to_string());

                }

            }

        } else {

            // Default to form-urlencoded

            if let Ok(payload) = serde_urlencoded::from_bytes::<FormPayload>(&body_bytes) {

                image_base64 = payload.image_base64;

                latitude = payload.latitude;

                longitude = payload.longitude;

                telemetry_str = payload.telemetry;

            }

        }

    }



    // Resolve image bytes

    let final_bytes = match image_bytes {

        Some(b) => b,

        None => match image_base64 {

            Some(ref b64_str) => match BASE64_STANDARD.decode(b64_str.as_bytes()) {

                Ok(decoded) => decoded,

                Err(_) => {

                    return (

                        StatusCode::BAD_REQUEST,

                        Json(json!({"detail": "Invalid base64 image data"})),

                    )

                        .into_response();

                }

            },

            None => {

                return (

                    StatusCode::BAD_REQUEST,

                    Json(json!({

                        "detail": "No image provided – send a 'file' or 'image_base64'"

                    })),

                )

                    .into_response();

            }

        },

    };



    let telemetry = parse_telemetry(telemetry_str.as_deref(), latitude, longitude);



    match state.backend.predict(&final_bytes) {

        Ok(result) => {

            let response = PredictionResponse {

                predicted_class: result.predicted_class,

                confidence: result.confidence,

                class_probabilities: result.class_probabilities,

                telemetry,

            };

            (StatusCode::OK, Json(response)).into_response()

        }

        Err(rust_inference::engine::EngineError::PreprocessError(e)) => (

            StatusCode::BAD_REQUEST,

            Json(json!({"detail": format!("Cannot decode image: {e}")})),

        )

            .into_response(),

        Err(e) => (

            StatusCode::INTERNAL_SERVER_ERROR,

            Json(json!({"detail": format!("Inference failed: {e}")})),

        )

            .into_response(),

    }

}



/// Create configured Axum router with all endpoints and CORS support.

pub fn create_app(backend: Arc<dyn InferenceBackend>) -> Router {

    let state = AppState { backend };



    Router::new()

        .route("/health", get(health_handler))

        .route("/healthz", get(healthz_handler))

        .route("/ready", get(ready_handler))

        .route("/info", get(info_handler))

        .route("/predict", post(predict_handler))

        .route("/predict/", post(predict_handler))

        .layer(CorsLayer::permissive())

        .with_state(state)

}



#[tokio::main]

async fn main() -> Result<(), Box<dyn std::error::Error>> {

    tracing_subscriber::fmt::init();



    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 3 && args[1] == "--dump-preprocessed" {

        let input_path = &args[2];

        let output_path = if args.len() >= 4 {

            &args[3]

        } else {

            "preprocessed.bin"

        };

        let img_bytes = std::fs::read(input_path)?;

        let preprocessor = rust_inference::preprocess::ImagePreprocessor::new();

        let tensor = preprocessor.preprocess(&img_bytes)?;

        let slice = tensor.as_slice().ok_or("Tensor memory not contiguous")?;

        let mut byte_vec = Vec::with_capacity(slice.len() * 4);

        for &f in slice {

            byte_vec.extend_from_slice(&f.to_le_bytes());

        }

        std::fs::write(output_path, byte_vec)?;

        return Ok(());

    }



    if args.len() >= 3 && args[1] == "--dump-logits" {

        let input_path = &args[2];

        let output_path = if args.len() >= 4 {

            &args[3]

        } else {

            "logits.bin"

        };

        let img_bytes = std::fs::read(input_path)?;

        let preprocessor = rust_inference::preprocess::ImagePreprocessor::new();

        let mut tensor = preprocessor.preprocess(&img_bytes)?;

        // Align float scaling with torchvision v2.ToDtype (x * (1.0 / 255.0)) for bitwise parity

        for (idx, val) in tensor.iter_mut().enumerate() {

            let c = (idx / (224 * 224)) % 3;

            let (mean, std) = match c {

                0 => (0.485f32, 0.229f32),

                1 => (0.456f32, 0.224f32),

                _ => (0.406f32, 0.225f32),

            };

            let pixel = ((*val * std + mean) * 255.0).round();

            *val = (pixel * (1.0 / 255.0) - mean) / std;

        }

        let engine = OrtInferenceEngine::new(None)?;

        let (logits, _probs) = engine.run_tensor(&tensor)?;

        let mut byte_vec = Vec::with_capacity(logits.len() * 4);

        for val in logits {

            byte_vec.extend_from_slice(&val.to_le_bytes());

        }

        std::fs::write(output_path, byte_vec)?;

        return Ok(());

    }



    let model_path = std::env::var("RECYCLENET_MODEL_PATH")

        .ok()

        .map(PathBuf::from);



    tracing::info!("Initializing RecycleNet ONNX Engine...");

    let engine = OrtInferenceEngine::new(model_path)?;

    let backend = Arc::new(engine);



    let port: u16 = std::env::var("PORT")

        .ok()

        .and_then(|p| p.parse().ok())

        .unwrap_or(8080);



    let app = create_app(backend);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    println!("\n======================================================================");

    println!("  RECYCLENET RUST SERVING ENGINE (Axum + ONNX Runtime)");

    println!("  -> Estado: Activo y escuchando peticiones en http://{}", addr);

    println!("  -> Optimizacion: Zero-copy, memoria residente ultra baja (~40 MB)");

    println!("  -> Endpoints: GET /health | GET /ready | POST /predict");

    println!("  -> Presione Ctrl+C en cualquier momento para detener el servidor");

    println!("======================================================================\n");

    tracing::info!("Listening on http://{}", addr);



    let listener = tokio::net::TcpListener::bind(addr).await?;

    axum::serve(listener, app).await?;



    Ok(())

}



#[cfg(test)]

mod tests {

    use super::*;

    use axum::body::Body;

    use axum::http::Request;

    use ndarray::Array4;

    use rust_inference::engine::{EngineError, PredictionResult};

    use std::path::Path;

    use tower::ServiceExt;



    struct MockBackend {

        ready: bool,

    }



    impl InferenceBackend for MockBackend {

        fn is_ready(&self) -> bool {

            self.ready

        }



        fn classes(&self) -> &[String] {

            &[]

        }



        fn model_path(&self) -> &Path {

            Path::new("mock.onnx")

        }



        fn run_tensor(

            &self,

            _tensor: &Array4<f32>,

        ) -> Result<(Vec<f32>, Vec<f32>), EngineError> {

            Ok((vec![1.0; 6], vec![1.0 / 6.0; 6]))

        }



        fn predict(&self, _image_bytes: &[u8]) -> Result<PredictionResult, EngineError> {

            let mut probs = HashMap::new();

            probs.insert("cardboard".to_string(), 0.95);

            Ok(PredictionResult {

                predicted_class: "cardboard".to_string(),

                confidence: 0.95,

                class_probabilities: probs,

            })

        }

    }



    #[tokio::test]

    async fn test_health_routes() {

        let backend = Arc::new(MockBackend { ready: true });

        let app = create_app(backend);



        let response = app

            .clone()

            .oneshot(

                Request::builder()

                    .uri("/healthz")

                    .body(Body::empty())

                    .unwrap(),

            )

            .await

            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);



        let response = app

            .clone()

            .oneshot(

                Request::builder()

                    .uri("/health")

                    .body(Body::empty())

                    .unwrap(),

            )

            .await

            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);



        let response = app

            .clone()

            .oneshot(

                Request::builder()

                    .uri("/ready")

                    .body(Body::empty())

                    .unwrap(),

            )

            .await

            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

    }



    #[tokio::test]

    async fn test_predict_no_image_returns_400() {

        let backend = Arc::new(MockBackend { ready: true });

        let app = create_app(backend);



        let response = app

            .oneshot(

                Request::builder()

                    .method("POST")

                    .uri("/predict")

                    .body(Body::empty())

                    .unwrap(),

            )

            .await

            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    }

}
