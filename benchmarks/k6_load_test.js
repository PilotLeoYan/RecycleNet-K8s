/**
 * RecycleNet-K8s: Production Load & Stress Testing Script (k6)
 *
 * Benchmarks both the Python FastAPI baseline (:8000) and the Rust Axum
 * high-performance engine (:8080) against strict Service Level Objectives (SLOs).
 *
 * Usage:
 *   # Python FastAPI baseline:
 *   k6 run -e TARGET_URL=http://localhost:8000 benchmarks/k6_load_test.js
 *
 *   # Rust Axum high-performance engine:
 *   k6 run -e TARGET_URL=http://localhost:8080 benchmarks/k6_load_test.js
 *
 *   # Output JSON summary or HTML report:
 *   k6 run --summary-export=k6-summary.json benchmarks/k6_load_test.js
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// Target base URL configurable via TARGET_URL environment variable (default: http://localhost:8000)
const TARGET_URL = __ENV.TARGET_URL || 'http://localhost:8000';

// Custom metric trends for fine-grained SLO observability
const inferenceLatencyTrend = new Trend('inference_latency_ms', true);
const healthzLatencyTrend = new Trend('healthz_latency_ms', true);
const infoLatencyTrend = new Trend('info_latency_ms', true);
const inferenceSuccessRate = new Rate('inference_success_rate');
const successfulPredictions = new Counter('successful_predictions');
const failedPredictions = new Counter('failed_predictions');

// Valid 32x32 base64-encoded JPEG image (compatible with Pillow and Rust image crate)
const SAMPLE_BASE64_JPEG =
  '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAgACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwC7RRRXunihRRRQAUUUUAFFFFAH/9k=';

export const options = {
  stages: [
    { duration: '15s', target: 10 }, // Stage 1: Warm-up (10 Virtual Users)
    { duration: '30s', target: 30 }, // Stage 2: Ramp-up (30 Virtual Users)
    { duration: '45s', target: 60 }, // Stage 3: Peak load (60 Virtual Users)
    { duration: '15s', target: 0 },  // Stage 4: Cooldown (0 Virtual Users)
  ],
  thresholds: {
    // Global SLO thresholds
    http_req_failed: ['rate<0.01'], // Total failure rate must be under 1%
    http_req_duration: ['p(95)<300', 'p(99)<600'], // Latency SLOs: p(95)<300ms, p(99)<600ms
    // Specific inference latency SLOs
    'inference_latency_ms': ['p(95)<300', 'p(99)<600'],
    'inference_success_rate': ['rate>0.99'],
  },
};

export default function () {
  // 1. Kubernetes Liveness Probe Verification (GET /healthz)
  group('Liveness Probe /healthz', function () {
    const res = http.get(`${TARGET_URL}/healthz`, {
      tags: { endpoint: 'healthz' },
    });
    healthzLatencyTrend.add(res.timings.duration);

    check(res, {
      'healthz status is 200': (r) => r.status === 200,
      'healthz returns ok status': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === 'ok';
        } catch (_) {
          return false;
        }
      },
    });
  });

  // 2. Service and Model Metadata Verification (GET /info)
  group('Service Metadata /info', function () {
    const res = http.get(`${TARGET_URL}/info`, {
      tags: { endpoint: 'info' },
    });
    infoLatencyTrend.add(res.timings.duration);

    check(res, {
      'info status is 200': (r) => r.status === 200,
      'info includes model_architecture': (r) => {
        try {
          const body = JSON.parse(r.body);
          return (
            body.model_architecture === 'MobileNetV3' ||
            typeof body.model_architecture === 'string'
          );
        } catch (_) {
          return false;
        }
      },
      'info confirms model_ready is true': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.model_ready === true;
        } catch (_) {
          return false;
        }
      },
      'info includes classes array': (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body.classes) && body.classes.length > 0;
        } catch (_) {
          return false;
        }
      },
    });
  });

  // 3. High-Throughput Waste Inference (POST /predict) with Base64 & GPS Telemetry
  group('Waste Inference /predict', function () {
    const payload = {
      image_base64: SAMPLE_BASE64_JPEG,
      latitude: '37.774929',
      longitude: '-122.419418',
    };

    const params = {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      tags: { endpoint: 'predict' },
    };

    const res = http.post(`${TARGET_URL}/predict`, payload, params);
    inferenceLatencyTrend.add(res.timings.duration);

    const isSuccess = check(res, {
      'predict status is 200': (r) => r.status === 200,
      'predict returns predicted_class': (r) => {
        try {
          const body = JSON.parse(r.body);
          return (
            typeof body.predicted_class === 'string' &&
            body.predicted_class.length > 0
          );
        } catch (_) {
          return false;
        }
      },
      'predict returns valid confidence score': (r) => {
        try {
          const body = JSON.parse(r.body);
          return (
            typeof body.confidence === 'number' &&
            body.confidence >= 0.0 &&
            body.confidence <= 1.0
          );
        } catch (_) {
          return false;
        }
      },
      'predict returns class probabilities map': (r) => {
        try {
          const body = JSON.parse(r.body);
          return (
            typeof body.class_probabilities === 'object' &&
            body.class_probabilities !== null &&
            Object.keys(body.class_probabilities).length > 0
          );
        } catch (_) {
          return false;
        }
      },
      'predict returns telemetry GPS coordinates': (r) => {
        try {
          const body = JSON.parse(r.body);
          return (
            body.telemetry !== null &&
            typeof body.telemetry === 'object' &&
            body.telemetry.latitude !== undefined &&
            body.telemetry.longitude !== undefined
          );
        } catch (_) {
          return false;
        }
      },
    });

    inferenceSuccessRate.add(isSuccess);
    if (isSuccess) {
      successfulPredictions.add(1);
    } else {
      failedPredictions.add(1);
    }
  });

  // Pacing interval between requests (50ms)
  sleep(0.05);
}
