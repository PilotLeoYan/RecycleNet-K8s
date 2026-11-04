"""Real-time webcam inference demonstration for RecycleNet.

Classifies waste materials in real-time using the local ONNX MobileNetV3 model,
drawing live probabilities, confidence levels, and an ROI target guide.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("recyclenet.webcam")

DEFAULT_MODEL_PATH = Path("models/recyclenet.onnx")
DEFAULT_CLASSES_PATH = Path("models/classes_mapping.json")

# ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Class color schemes (BGR)
CLASS_COLORS = {
    "cardboard": (60, 140, 200),  # Brown / Ochre
    "glass": (210, 210, 80),  # Cyan / Light blue
    "metal": (180, 180, 180),  # Metallic gray
    "paper": (230, 230, 230),  # White / Soft gray
    "plastic": (40, 210, 120),  # Vibrant green
    "trash": (80, 80, 220),  # Red / Hazard
}


class WasteClassifier:
    """Manages the local ONNX inference session and preprocessing."""

    def __init__(self, model_path: Path, classes_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not classes_path.exists():
            raise FileNotFoundError(f"Classes mapping file not found: {classes_path}")

        with open(classes_path, encoding="utf-8") as f:
            raw_classes = json.load(f)
            self.classes = {int(k): v for k, v in raw_classes.items()}

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, roi_bgr: np.ndarray) -> tuple[int, str, float, np.ndarray, float]:
        """Preprocesses the ROI crop, executes ONNX inference, and returns scores.

        Returns:
            (class_idx, class_name, confidence, all_probabilities, latency_ms)
        """
        t0 = time.perf_counter()

        # Convert BGR -> RGB and resize to (224, 224)
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Normalize with ImageNet mean and std
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        blob = np.transpose(normalized, (2, 0, 1))[np.newaxis, :, :, :]

        # Forward pass
        logits = self.session.run(None, {self.input_name: blob})[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        predicted_idx = int(np.argmax(probabilities))
        predicted_name = self.classes.get(predicted_idx, "unknown")
        confidence = float(probabilities[predicted_idx])

        return predicted_idx, predicted_name, confidence, probabilities, latency_ms


def draw_hud(
    frame: np.ndarray,
    roi_coords: tuple[int, int, int, int],
    predicted_class: str,
    confidence: float,
    probabilities: np.ndarray,
    classes: dict[int, str],
    fps: float,
    latency_ms: float,
    paused: bool,
) -> None:
    """Renders real-time HUD, targeting box, class banners, and probability bars."""
    x1, y1, x2, y2 = roi_coords
    color = CLASS_COLORS.get(predicted_class, (0, 255, 0))

    # 1. Draw central ROI target box with corner accents
    corner_len = 25
    thickness = 2
    # Bounding box border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

    # Accent corners
    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    # Center crosshair guide (subtle)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (120, 120, 120), 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (120, 120, 120), 1)

    # 2. Top Banner with Prediction
    banner_h = 55
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    status_tag = "[PAUSADO] " if paused else ""
    title_text = f"{status_tag}RECYCLENET: {predicted_class.upper()}"
    conf_text = f"{confidence * 100:.1f}% CONF"

    cv2.putText(
        frame,
        title_text,
        (15, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        conf_text,
        (frame.shape[1] - 180, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # 3. Probability Breakdown Sidebar (Left side)
    side_x = 15
    side_y = 80
    row_height = 24
    bar_width = 110

    # Background for sidebar
    sidebar_w = 230
    sidebar_h = len(classes) * row_height + 35
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (side_x - 5, side_y - 18),
        (side_x + sidebar_w, side_y + sidebar_h),
        (20, 20, 25),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        "PROBABILIDADES",
        (side_x, side_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    for i, c_name in classes.items():
        prob = float(probabilities[i]) if i < len(probabilities) else 0.0
        y_pos = side_y + 18 + (i * row_height)
        is_winner = c_name == predicted_class
        label_color = (255, 255, 255) if is_winner else (160, 160, 160)

        # Class name
        cv2.putText(
            frame,
            f"{c_name:<10}",
            (side_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            label_color,
            1,
            cv2.LINE_AA,
        )

        # Bar background
        bar_x = side_x + 85
        cv2.rectangle(
            frame,
            (bar_x, y_pos - 10),
            (bar_x + bar_width, y_pos + 1),
            (60, 60, 60),
            -1,
        )

        # Active fill bar
        fill_w = int(bar_width * prob)
        bar_color = CLASS_COLORS.get(c_name, (100, 200, 100))
        if fill_w > 0:
            cv2.rectangle(
                frame,
                (bar_x, y_pos - 10),
                (bar_x + fill_w, y_pos + 1),
                bar_color,
                -1,
            )

        # Numeric %
        cv2.putText(
            frame,
            f"{prob * 100:4.1f}%",
            (bar_x + bar_width + 8, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            label_color,
            1,
            cv2.LINE_AA,
        )

    # 4. Bottom Footer: Performance & Hotkeys
    footer_y = frame.shape[0] - 12
    device_info = f"FPS: {fps:4.1f} | Latencia: {latency_ms:4.1f} ms | CPU (ONNX)"
    cv2.putText(
        frame,
        device_info,
        (15, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    hotkeys_text = "[ESPACIO]: Pausar | [S]: Captura | [Q]: Salir"
    cv2.putText(
        frame,
        hotkeys_text,
        (frame.shape[1] - 310, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )


def run_webcam_demo(
    camera_index: int = 0,
    model_path: Path = DEFAULT_MODEL_PATH,
    classes_path: Path = DEFAULT_CLASSES_PATH,
    smooth_factor: float = 0.35,
) -> None:
    """Executes the real-time webcam classification loop."""
    print("=" * 65)
    print("       RECYCLENET - CLASIFICADOR EN TIEMPO REAL (WEBCAM)")
    print("=" * 65)
    print(f"[*] Cargando modelo ONNX: {model_path}")
    classifier = WasteClassifier(model_path, classes_path)
    print(f"[OK] Modelo cargado con {len(classifier.classes)} clases detectadas.")

    print(f"[*] Iniciando camara web (indice: {camera_index})...")
    # On Windows, cv2.CAP_DSHOW can speed up DirectShow camera initialization
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.warning(
            "No se pudo abrir con DirectShow; reintentando backend predeterminado..."
        )
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERR] No se pudo acceder a la camara web en el indice {camera_index}.")
        print("      Verifica que este conectada o prueba pasando --camera 1")
        return

    # Attempt standard 720p or 480p resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "RecycleNet · Clasificacion de Residuos en Vivo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[+] Camara iniciada con exito.")
    print("    -> Coloca el objeto en el recuadro central.")
    print("    -> [ESPACIO]: Pausar/Reanudar inferencia")
    print("    -> [S]: Guardar fotograma en disco")
    print("    -> [Q] o [ESC]: Cerrar demostracion\n")

    smoothed_probs: np.ndarray | None = None
    last_class = "Iniciando..."
    last_conf = 0.0
    latency_ms = 0.0
    fps = 0.0
    fps_time = time.perf_counter()
    fps_counter = 0
    paused = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error("No se pudo recibir fotograma de la camara.")
                break

            # Horizontal flip for intuitive mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Compute central square ROI
            roi_size = int(min(h, w) * 0.65)
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            roi = frame[y1:y2, x1:x2]

            # Run inference if not paused
            if not paused and roi.size > 0:
                _, pred_class, _, raw_probs, latency_ms = classifier.predict(roi)

                # Exponential Moving Average for visual stability
                if smoothed_probs is None:
                    smoothed_probs = raw_probs
                else:
                    smoothed_probs = (
                        smooth_factor * raw_probs
                        + (1.0 - smooth_factor) * smoothed_probs
                    )

                top_idx = int(np.argmax(smoothed_probs))
                last_class = classifier.classes[top_idx]
                last_conf = float(smoothed_probs[top_idx])

            # Update FPS counter
            fps_counter += 1
            now = time.perf_counter()
            elapsed = now - fps_time
            if elapsed >= 0.5:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_time = now

            # Render overlay HUD
            display_probs = (
                smoothed_probs
                if smoothed_probs is not None
                else np.zeros(len(classifier.classes))
            )
            draw_hud(
                frame=frame,
                roi_coords=(x1, y1, x2, y2),
                predicted_class=last_class,
                confidence=last_conf,
                probabilities=display_probs,
                classes=classifier.classes,
                fps=fps,
                latency_ms=latency_ms,
                paused=paused,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                print("[*] Saliendo de la demostracion...")
                break
            if key == ord(" "):
                paused = not paused
                print(f"[*] Inferencia {'PAUSADA' if paused else 'REANUDADA'}")
            elif key in (ord("s"), ord("S")):
                capture_dir = Path("data/captures")
                capture_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time())
                save_path = capture_dir / f"capture_{last_class}_{timestamp}.jpg"
                cv2.imwrite(str(save_path), frame)
                print(f"[+] Fotograma guardado en: {save_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Camara liberada y ventanas cerradas.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RecycleNet Real-Time Webcam Classifier"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indice de la camara web (por defecto: 0)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta al modelo ONNX",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=DEFAULT_CLASSES_PATH,
        help="Ruta al mapeo de clases JSON",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.35,
        help="Factor de suavizado temporal EMA (0.1 a 1.0)",
    )

    args = parser.parse_args()
    run_webcam_demo(
        camera_index=args.camera,
        model_path=args.model,
        classes_path=args.classes,
        smooth_factor=args.smooth,
    )


if __name__ == "__main__":
    main()
