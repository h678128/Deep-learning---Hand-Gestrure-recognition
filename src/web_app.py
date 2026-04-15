from __future__ import annotations

import argparse
import base64
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template_string, request

from dataset import decode_heatmaps, infer_connections
from model import create_heatmap_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "modell"
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "web_app" / "uploads"
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Hand Landmark Demo</title>
    <style>
      :root {
        --bg: #f4efe6;
        --panel: #fff8ee;
        --ink: #1d1d1d;
        --accent: #0e8f5b;
        --accent-soft: #d7f1e6;
        --border: #d8c9b4;
      }
      body {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, #fff7cc 0, transparent 30%),
          linear-gradient(135deg, #f4efe6, #efe1cf);
      }
      .page {
        max-width: 1100px;
        margin: 0 auto;
        padding: 32px 20px 56px;
      }
      .hero {
        display: grid;
        gap: 14px;
        margin-bottom: 24px;
      }
      h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 0.95;
        letter-spacing: -0.03em;
      }
      .lead {
        max-width: 720px;
        font-size: 1.05rem;
        line-height: 1.5;
      }
      .panel {
        background: color-mix(in srgb, var(--panel) 92%, white);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(48, 32, 10, 0.08);
        padding: 18px;
      }
      form {
        display: grid;
        gap: 14px;
      }
      .grid {
        display: grid;
        gap: 14px;
      }
      @media (min-width: 760px) {
        .grid {
          grid-template-columns: 1.3fr 1fr 1fr;
          align-items: end;
        }
      }
      label {
        display: grid;
        gap: 8px;
        font-weight: 600;
      }
      input, select, button {
        font: inherit;
      }
      input[type="file"], select {
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px 12px;
      }
      button {
        border: none;
        border-radius: 999px;
        padding: 12px 18px;
        background: linear-gradient(135deg, #0e8f5b, #0a6b44);
        color: white;
        font-weight: 700;
        cursor: pointer;
      }
      .results {
        display: grid;
        gap: 18px;
        margin-top: 24px;
      }
      @media (min-width: 860px) {
        .results {
          grid-template-columns: 1fr 1fr;
        }
      }
      .image-card {
        display: grid;
        gap: 10px;
      }
      .image-card img {
        width: 100%;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: #f5f5f5;
      }
      .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 12px;
      }
      .pill {
        background: var(--accent-soft);
        color: #0c5a3d;
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 0.95rem;
        font-weight: 600;
      }
      .error {
        color: #9b1f1f;
        font-weight: 700;
      }
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <h1>Hand Landmark Demo</h1>
        <div class="lead">
          Upload an image and run one of the trained landmark models in the browser-backed demo.
          This is useful for showing unseen data in a more presentable way than the raw terminal scripts.
        </div>
      </section>

      <section class="panel">
        <form method="post" enctype="multipart/form-data">
          <div class="grid">
            <label>
              Image
              <input type="file" name="image" accept=".jpg,.jpeg,.png,.bmp,.webp" required>
            </label>
            <label>
              Checkpoint
              <select name="checkpoint">
                {% for checkpoint_name in checkpoint_names %}
                <option value="{{ checkpoint_name }}" {% if checkpoint_name == selected_checkpoint %}selected{% endif %}>
                  {{ checkpoint_name }}
                </option>
                {% endfor %}
              </select>
            </label>
            <label>
              Confidence Threshold
              <select name="confidence_threshold">
                {% for value in ["0.15", "0.20", "0.25", "0.30"] %}
                <option value="{{ value }}" {% if value == selected_threshold %}selected{% endif %}>
                  {{ value }}
                </option>
                {% endfor %}
              </select>
            </label>
          </div>
          <button type="submit">Run Prediction</button>
        </form>
        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}
      </section>

      {% if result_image %}
      <section class="results">
        <article class="panel image-card">
          <strong>Original</strong>
          <img src="data:image/jpeg;base64,{{ original_image }}" alt="Original image">
        </article>
        <article class="panel image-card">
          <strong>Prediction</strong>
          <img src="data:image/jpeg;base64,{{ result_image }}" alt="Prediction image">
          <div class="meta">
            <span class="pill">avg confidence: {{ avg_confidence }}</span>
            <span class="pill">min confidence: {{ min_confidence }}</span>
            <span class="pill">max confidence: {{ max_confidence }}</span>
          </div>
        </article>
      </section>
      {% endif %}
    </main>
  </body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small local web app for hand landmark inference.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--default-checkpoint",
        type=Path,
        default=MODEL_DIR / "landmark_heatmap11_augstrong20k.pt",
    )
    return parser.parse_args()


def available_checkpoints() -> list[Path]:
    checkpoints = sorted(MODEL_DIR.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError("No .pt checkpoints found in modell/")
    return checkpoints


def encode_image(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("Could not encode image for browser output.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def preprocess_image(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (image_size, image_size))
    return torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def upscale_landmarks(
    landmarks_xy: np.ndarray,
    source_size: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    target_h, target_w = target_hw
    scaled = landmarks_xy.copy()
    scaled[:, 0] *= target_w / float(source_size)
    scaled[:, 1] *= target_h / float(source_size)
    return scaled


def heatmap_confidences(heatmaps: torch.Tensor) -> np.ndarray:
    flattened = heatmaps.view(heatmaps.shape[0], heatmaps.shape[1], -1)
    return flattened.amax(dim=-1)[0].cpu().numpy()


def draw_prediction(
    image_bgr: np.ndarray,
    landmarks_xy: np.ndarray,
    connections: list[tuple[int, int]],
    confidences: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    canvas = image_bgr.copy()
    visible = confidences >= confidence_threshold

    for start_idx, end_idx in connections:
        if not (visible[start_idx] and visible[end_idx]):
            continue
        start_point = tuple(np.round(landmarks_xy[start_idx]).astype(int))
        end_point = tuple(np.round(landmarks_xy[end_idx]).astype(int))
        cv2.line(canvas, start_point, end_point, (0, 220, 120), 2)

    for index, point in enumerate(landmarks_xy):
        point_xy = tuple(np.round(point).astype(int))
        color = (0, 220, 120) if visible[index] else (0, 140, 255)
        cv2.circle(canvas, point_xy, 5, color, -1)

    cv2.putText(
        canvas,
        f"avg conf: {confidences.mean():.2f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


class CheckpointRegistry:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cache: dict[Path, tuple[dict[str, object], torch.nn.Module, list[tuple[int, int]]]] = {}

    def load(self, checkpoint_path: Path) -> tuple[dict[str, object], torch.nn.Module, list[tuple[int, int]]]:
        if checkpoint_path not in self.cache:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model = create_heatmap_model(num_landmarks=checkpoint["num_landmarks"]).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            connections = infer_connections(checkpoint.get("selected_landmark_indices"))
            self.cache[checkpoint_path] = (checkpoint, model, connections)
        return self.cache[checkpoint_path]


def create_app(default_checkpoint: Path) -> Flask:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = available_checkpoints()
    checkpoint_map = {path.name: path for path in checkpoint_paths}
    registry = CheckpointRegistry(device)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @app.route("/", methods=["GET", "POST"])
    def index():
        error = None
        result_image = None
        original_image = None
        avg_confidence = None
        min_confidence = None
        max_confidence = None

        selected_checkpoint = default_checkpoint.name if default_checkpoint.name in checkpoint_map else checkpoint_paths[0].name
        selected_threshold = "0.25"

        if request.method == "POST":
            selected_checkpoint = request.form.get("checkpoint", selected_checkpoint)
            selected_threshold = request.form.get("confidence_threshold", selected_threshold)
            uploaded_file = request.files.get("image")

            if uploaded_file is None or uploaded_file.filename == "":
                error = "Please choose an image before running prediction."
            elif selected_checkpoint not in checkpoint_map:
                error = "Selected checkpoint does not exist."
            else:
                image_bytes = uploaded_file.read()
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if image_bgr is None:
                    error = "Could not read the uploaded image."
                else:
                    checkpoint_path = checkpoint_map[selected_checkpoint]
                    checkpoint, model, connections = registry.load(checkpoint_path)
                    image_size = int(checkpoint.get("image_size", 224))
                    confidence_threshold = float(selected_threshold)

                    input_tensor = preprocess_image(image_bgr, image_size).to(device)
                    with torch.no_grad():
                        predicted_heatmaps = model(input_tensor)
                        predicted_landmarks = decode_heatmaps(
                            predicted_heatmaps.cpu(),
                            image_size=image_size,
                        )[0].numpy()
                        confidences = heatmap_confidences(predicted_heatmaps.cpu())

                    predicted_landmarks = upscale_landmarks(
                        predicted_landmarks,
                        source_size=image_size,
                        target_hw=image_bgr.shape[:2],
                    )
                    preview_bgr = draw_prediction(
                        image_bgr,
                        predicted_landmarks,
                        connections,
                        confidences,
                        confidence_threshold=confidence_threshold,
                    )

                    original_image = encode_image(image_bgr)
                    result_image = encode_image(preview_bgr)
                    avg_confidence = f"{float(confidences.mean()):.3f}"
                    min_confidence = f"{float(confidences.min()):.3f}"
                    max_confidence = f"{float(confidences.max()):.3f}"

        return render_template_string(
            HTML_TEMPLATE,
            checkpoint_names=sorted(checkpoint_map),
            selected_checkpoint=selected_checkpoint,
            selected_threshold=selected_threshold,
            error=error,
            original_image=original_image,
            result_image=result_image,
            avg_confidence=avg_confidence,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
        )

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args.default_checkpoint)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
