from __future__ import annotations

import argparse
import base64
import importlib
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request

from dataset import DEFAULT_LANDMARK_INDICES, decode_heatmaps, infer_connections
from model import create_heatmap_model_from_checkpoint

try:
    import mediapipe as mp
except ImportError:
    mp = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "modell"
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "web_app" / "uploads"
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Hand Landmark Demo</title>
    <style>
      :root {
        --bg: #efe6d5;
        --bg-deep: #dcc6a3;
        --panel: rgba(255, 249, 239, 0.9);
        --ink: #1f1a14;
        --muted: #5f5346;
        --accent: #126b57;
        --accent-soft: #d8f0e6;
        --accent-warm: #bf6f34;
        --border: rgba(111, 85, 50, 0.18);
        --shadow: 0 18px 40px rgba(54, 34, 8, 0.12);
      }
      body {
        margin: 0;
        font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(255, 247, 204, 0.95) 0, transparent 28%),
          radial-gradient(circle at bottom right, rgba(18, 107, 87, 0.12) 0, transparent 24%),
          linear-gradient(145deg, var(--bg), var(--bg-deep));
        min-height: 100vh;
      }
      .page {
        max-width: 1180px;
        margin: 0 auto;
        padding: 28px 20px 64px;
      }
      .hero {
        display: grid;
        gap: 18px;
        margin-bottom: 28px;
        padding: 26px;
        background:
          linear-gradient(135deg, rgba(255, 250, 242, 0.96), rgba(245, 235, 218, 0.92)),
          repeating-linear-gradient(
            135deg,
            rgba(111, 85, 50, 0.03) 0,
            rgba(111, 85, 50, 0.03) 10px,
            transparent 10px,
            transparent 22px
          );
        border: 1px solid var(--border);
        border-radius: 28px;
        box-shadow: var(--shadow);
      }
      h1 {
        margin: 0;
        font-size: clamp(2.4rem, 5vw, 4.8rem);
        line-height: 0.9;
        letter-spacing: -0.03em;
        max-width: 9ch;
      }
      .lead {
        max-width: 760px;
        font-size: 1.08rem;
        line-height: 1.6;
        color: var(--muted);
      }
      .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        padding: 9px 14px;
        background: rgba(18, 107, 87, 0.08);
        color: #134f42;
        font-size: 0.95rem;
        font-weight: 700;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: var(--shadow);
        padding: 20px;
        backdrop-filter: blur(10px);
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
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 12px 14px;
        color: var(--ink);
      }
      button {
        border: none;
        border-radius: 999px;
        padding: 12px 18px;
        background: linear-gradient(135deg, var(--accent), #0d5343);
        color: white;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 10px 22px rgba(18, 107, 87, 0.25);
        transition: transform 120ms ease, box-shadow 120ms ease;
      }
      button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(18, 107, 87, 0.32);
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
      .split {
        display: grid;
        gap: 18px;
      }
      @media (min-width: 900px) {
        .split {
          grid-template-columns: 1.1fr 0.9fr;
        }
      }
      .image-card {
        display: grid;
        gap: 10px;
      }
      .image-card img, .image-card video, .image-card canvas {
        width: 100%;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: #f5f5f5;
      }
      .camera-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .secondary {
        background: rgba(255, 255, 255, 0.84);
        color: var(--ink);
        border: 1px solid var(--border);
        box-shadow: none;
      }
      .status {
        font-size: 0.95rem;
        color: var(--muted);
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255,255,255,0.6);
        border: 1px solid rgba(111, 85, 50, 0.1);
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
        margin: 0;
      }
      .section-kicker {
        display: inline-block;
        margin-bottom: 8px;
        color: var(--accent-warm);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }
      .section-title {
        margin: 0 0 6px;
        font-size: 1.5rem;
        line-height: 1.1;
      }
      .section-copy {
        margin: 0 0 16px;
        color: var(--muted);
        line-height: 1.55;
      }
      .camera-frame {
        position: relative;
      }
      .camera-frame::after {
        content: "";
        position: absolute;
        inset: 14px;
        border: 1px dashed rgba(255,255,255,0.45);
        border-radius: 16px;
        pointer-events: none;
      }
      /* Modern demo layout override: live camera is the primary interaction. */
      :root {
        --bg: #f5f3ee;
        --bg-deep: #ebe7df;
        --panel: rgba(255, 255, 255, 0.82);
        --ink: #171717;
        --muted: #66707a;
        --accent: #0b6b5b;
        --accent-soft: #dcefe9;
        --accent-warm: #b86031;
        --border: rgba(23, 23, 23, 0.10);
        --shadow: 0 24px 70px rgba(20, 24, 28, 0.10);
      }
      * {
        box-sizing: border-box;
      }
      body {
        font-family: "Aptos Display", "Segoe UI Variable", "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at 10% 0%, rgba(11, 107, 91, 0.14), transparent 32%),
          radial-gradient(circle at 88% 10%, rgba(184, 96, 49, 0.10), transparent 28%),
          linear-gradient(180deg, #fbfaf7, var(--bg));
      }
      .page {
        max-width: 1240px;
        padding-top: 24px;
      }
      .hero {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 24px;
        margin-bottom: 22px;
        padding: 8px 2px 12px;
        background: transparent;
        border: 0;
        border-radius: 0;
        box-shadow: none;
      }
      h1 {
        font-size: clamp(2.4rem, 6vw, 5.8rem);
        line-height: 0.88;
        letter-spacing: -0.07em;
        max-width: 11ch;
      }
      .lead {
        max-width: 560px;
        margin-top: 14px;
        font-size: 1rem;
      }
      .hero-meta {
        justify-content: flex-end;
        gap: 8px;
      }
      .hero-pill {
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid var(--border);
        color: var(--muted);
        font-size: 0.86rem;
        letter-spacing: 0.01em;
      }
      .panel {
        border-radius: 28px;
        padding: 18px;
        backdrop-filter: blur(18px);
      }
      .split {
        grid-template-columns: minmax(0, 1.55fr) minmax(300px, 0.7fr);
        align-items: stretch;
      }
      .split > .image-card {
        order: -1;
      }
      .split > .image-card.panel {
        background: rgba(13, 17, 16, 0.94);
        color: white;
      }
      .split > .image-card .section-copy {
        color: rgba(255, 255, 255, 0.68);
      }
      .split > .image-card .section-kicker {
        color: #94e3d1;
      }
      .camera-frame {
        overflow: hidden;
        border-radius: 22px;
        background: #0d1110;
        min-height: 460px;
      }
      .camera-frame video {
        width: 100%;
        height: 100%;
        min-height: 460px;
        object-fit: cover;
        display: block;
        border: 1px solid rgba(255, 255, 255, 0.12);
      }
      .camera-actions {
        margin-top: 4px;
      }
      .camera-actions button {
        flex: 1 1 160px;
      }
      input[type="file"], select {
        width: 100%;
        border-radius: 16px;
        border-color: var(--border);
        background: rgba(255, 255, 255, 0.92);
      }
      button {
        padding: 13px 18px;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent), #084b41);
      }
      .secondary {
        background: rgba(255, 255, 255, 0.92);
      }
      .status {
        margin-top: 8px;
        border-radius: 16px;
      }
      .results {
        margin-top: 18px;
      }
      .section-title {
        letter-spacing: -0.03em;
      }
      .live-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
        gap: 18px;
        margin-bottom: 18px;
      }
      .live-results {
        display: grid;
        gap: 18px;
      }
      .live-results .panel {
        min-height: 0;
      }
      .live-results .image-card img {
        max-height: 310px;
        object-fit: contain;
      }
      .live-placeholder {
        min-height: 250px;
        display: grid;
        place-items: center;
        border-radius: 18px;
        border: 1px dashed var(--border);
        color: var(--muted);
        text-align: center;
        padding: 22px;
        background: rgba(255, 255, 255, 0.58);
      }
      @media (max-width: 960px) {
        .hero {
          align-items: flex-start;
          flex-direction: column;
        }
        .hero-meta {
          justify-content: flex-start;
        }
        .split {
          grid-template-columns: 1fr;
        }
        .live-grid {
          grid-template-columns: 1fr;
        }
        .camera-frame, .camera-frame video {
          min-height: 320px;
        }
      }
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <div class="hero-meta">
          <span class="hero-pill">11 landmarks</span>
          <span class="hero-pill">MediaPipe ROI</span>
          <span class="hero-pill">Live camera</span>
        </div>
        <h1>Live Hand Tracking</h1>
        <div class="lead">
          Start the webcam to run the trained heatmap model on live hand input. You can also capture a single
          camera frame or upload an image for controlled testing.
        </div>
      </section>

      <section class="live-grid">
        <section class="split">
          <section class="panel">
            <div class="section-kicker">Model And Image Test</div>
            <h2 class="section-title">Choose model and test an image</h2>
            <p class="section-copy">
              The selected checkpoint and threshold are used for live camera, snapshots, and uploaded images.
            </p>
            <form method="post" enctype="multipart/form-data">
              <div class="grid">
                <label>
                  Image
                  <input type="file" name="image" accept=".jpg,.jpeg,.png,.bmp,.webp" required>
                </label>
                <label>
                  Checkpoint
                  <select name="checkpoint" id="checkpoint-select">
                    {% for checkpoint_name in checkpoint_names %}
                    <option value="{{ checkpoint_name }}" {% if checkpoint_name == selected_checkpoint %}selected{% endif %}>
                      {{ checkpoint_name }}
                    </option>
                    {% endfor %}
                  </select>
                </label>
                <label>
                  Confidence threshold
                  <select name="confidence_threshold" id="confidence-threshold">
                    {% for value in ["0.10", "0.15", "0.20", "0.25", "0.30", "0.35"] %}
                    <option value="{{ value }}" {% if value == selected_threshold %}selected{% endif %}>
                      {{ value }}
                    </option>
                    {% endfor %}
                  </select>
                </label>
              </div>
              <button type="submit">Predict uploaded image</button>
            </form>
            {% if error %}
            <p class="error">{{ error }}</p>
            {% endif %}
          </section>

          <section class="panel image-card">
            <div class="section-kicker">Primary Demo</div>
            <h2 class="section-title">Use your webcam live</h2>
            <p class="section-copy">
              Give the browser access to your camera, then run live prediction or capture one frame for a still result.
            </p>
            <div class="camera-frame">
              <video id="camera-preview" autoplay playsinline muted></video>
            </div>
            <canvas id="camera-canvas" hidden></canvas>
            <div class="camera-actions">
              <button type="button" id="start-camera">Start camera</button>
              <button type="button" class="secondary" id="capture-frame">Take picture</button>
              <button type="button" class="secondary" id="toggle-live">Start live</button>
            </div>
            <div class="status" id="camera-status">
              Camera is idle. Click "Start camera" to allow browser access.
            </div>
          </section>
        </section>

        <section class="live-results" id="camera-results">
          <article class="panel image-card">
            <div class="section-kicker">Live Output</div>
            <strong>Model prediction</strong>
            <img id="camera-prediction" src="{% if result_image %}data:image/jpeg;base64,{{ result_image }}{% endif %}" alt="Camera prediction">
            <div class="live-placeholder" id="camera-placeholder" {% if result_image %}style="display:none"{% endif %}>
              Start live prediction to see the landmark overlay here.
            </div>
            <div class="meta">
              <span class="pill" id="camera-avg">avg confidence: {{ avg_confidence or "-" }}</span>
              <span class="pill" id="camera-min">min confidence: {{ min_confidence or "-" }}</span>
              <span class="pill" id="camera-max">max confidence: {{ max_confidence or "-" }}</span>
            </div>
          </article>
          <article class="panel image-card">
            <div class="section-kicker">Camera Snapshot</div>
            <strong>Input frame</strong>
            <img id="camera-original" src="{% if original_image %}data:image/jpeg;base64,{{ original_image }}{% endif %}" alt="Camera snapshot">
          </article>
        </section>
      </section>

      {% if result_image %}
      <section class="results">
        <article class="panel image-card">
          <div class="section-kicker">Input</div>
          <strong>Original Image</strong>
          <img src="data:image/jpeg;base64,{{ original_image }}" alt="Original image">
        </article>
        <article class="panel image-card">
          <div class="section-kicker">Output</div>
          <strong>Predicted Landmarks</strong>
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
    <script>
      const startButton = document.getElementById("start-camera");
      const captureButton = document.getElementById("capture-frame");
      const liveButton = document.getElementById("toggle-live");
      const video = document.getElementById("camera-preview");
      const canvas = document.getElementById("camera-canvas");
      const statusText = document.getElementById("camera-status");
      const checkpointSelect = document.getElementById("checkpoint-select");
      const thresholdSelect = document.getElementById("confidence-threshold");
      const resultsSection = document.getElementById("camera-results");
      const cameraOriginal = document.getElementById("camera-original");
      const cameraPrediction = document.getElementById("camera-prediction");
      const cameraAvg = document.getElementById("camera-avg");
      const cameraMin = document.getElementById("camera-min");
      const cameraMax = document.getElementById("camera-max");
      const cameraPlaceholder = document.getElementById("camera-placeholder");

      let mediaStream = null;
      let liveInterval = null;
      let isSending = false;

      async function ensureCamera() {
        if (mediaStream) {
          return true;
        }
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          video.srcObject = mediaStream;
          statusText.textContent = "Camera ready.";
          return true;
        } catch (error) {
          statusText.textContent = "Could not access camera. Check browser permissions.";
          return false;
        }
      }

      function snapshotDataUrl() {
        const width = video.videoWidth;
        const height = video.videoHeight;
        if (!width || !height) {
          return null;
        }
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, width, height);
        return canvas.toDataURL("image/jpeg", 0.9);
      }

      async function runPrediction() {
        if (isSending) {
          return;
        }
        const ready = await ensureCamera();
        if (!ready) {
          return;
        }
        const imageData = snapshotDataUrl();
        if (!imageData) {
          statusText.textContent = "Camera is starting. Try again in a second.";
          return;
        }

        isSending = true;
        statusText.textContent = liveInterval ? "Running live prediction..." : "Running snapshot prediction...";

        try {
          const response = await fetch("/predict-api", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              image_data: imageData,
              checkpoint: checkpointSelect.value,
              confidence_threshold: thresholdSelect.value
            })
          });
          const payload = await response.json();
          if (!response.ok) {
            statusText.textContent = payload.error || "Prediction failed.";
            return;
          }

          resultsSection.style.display = "grid";
          cameraOriginal.src = payload.original_image;
          cameraPrediction.src = payload.result_image;
          cameraPrediction.style.display = "block";
          if (cameraPlaceholder) {
            cameraPlaceholder.style.display = "none";
          }
          cameraAvg.textContent = `avg confidence: ${payload.avg_confidence}`;
          cameraMin.textContent = `min confidence: ${payload.min_confidence}`;
          cameraMax.textContent = `max confidence: ${payload.max_confidence}`;
          statusText.textContent = liveInterval ? "Live prediction running." : "Snapshot prediction complete.";
        } catch (error) {
          statusText.textContent = "Prediction request failed.";
        } finally {
          isSending = false;
        }
      }

      startButton.addEventListener("click", async () => {
        await ensureCamera();
      });

      captureButton.addEventListener("click", async () => {
        await runPrediction();
      });

      liveButton.addEventListener("click", async () => {
        if (liveInterval) {
          clearInterval(liveInterval);
          liveInterval = null;
          liveButton.textContent = "Start Live";
          statusText.textContent = "Live prediction stopped.";
          return;
        }

        const ready = await ensureCamera();
        if (!ready) {
          return;
        }

        liveButton.textContent = "Stop Live";
        statusText.textContent = "Live prediction starting...";
        liveInterval = setInterval(runPrediction, 700);
        runPrediction();
      });
    </script>
  </body>
</html>
"""


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Live Hand Tracking</title>
    <style>
      :root {
        --bg: #f6f4ee;
        --surface: rgba(255, 255, 255, 0.82);
        --surface-strong: #ffffff;
        --ink: #151716;
        --muted: #687178;
        --line: rgba(21, 23, 22, 0.10);
        --accent: #0b6b5b;
        --accent-dark: #07483e;
        --accent-soft: #dcefe9;
        --warm: #b85f33;
        --shadow: 0 24px 70px rgba(24, 31, 35, 0.12);
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        min-height: 100vh;
        color: var(--ink);
        font-family: "Aptos Display", "Segoe UI Variable", "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at 12% 0%, rgba(11, 107, 91, 0.16), transparent 30%),
          radial-gradient(circle at 88% 8%, rgba(184, 95, 51, 0.11), transparent 28%),
          linear-gradient(180deg, #fbfaf7 0%, var(--bg) 100%);
      }
      button, input, select {
        font: inherit;
      }
      .page {
        width: min(1480px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 20px 0 40px;
      }
      .topbar {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 24px;
        margin-bottom: 16px;
      }
      .eyebrow {
        margin: 0 0 8px;
        color: var(--warm);
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.15em;
        text-transform: uppercase;
      }
      h1 {
        max-width: 680px;
        margin: 0;
        font-size: clamp(2.2rem, 5vw, 5.2rem);
        line-height: 0.88;
        letter-spacing: -0.07em;
      }
      .lead {
        max-width: 520px;
        margin: 0;
        color: var(--muted);
        line-height: 1.55;
      }
      .badges {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 8px;
      }
      .badge, .metric {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 8px 11px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.86rem;
        font-weight: 760;
      }
      .app-shell {
        display: grid;
        grid-template-columns: minmax(250px, 320px) minmax(360px, 1.15fr) minmax(360px, 1fr);
        gap: 16px;
        align-items: stretch;
      }
      .panel {
        border: 1px solid var(--line);
        border-radius: 28px;
        background: var(--surface);
        box-shadow: var(--shadow);
        backdrop-filter: blur(18px);
      }
      .controls {
        display: grid;
        align-content: start;
        gap: 14px;
        padding: 18px;
      }
      .panel-title {
        margin: 0;
        font-size: 1.35rem;
        line-height: 1.1;
        letter-spacing: -0.04em;
      }
      .panel-copy {
        margin: 7px 0 2px;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.48;
      }
      label {
        display: grid;
        gap: 7px;
        color: #323737;
        font-size: 0.88rem;
        font-weight: 760;
      }
      select, input[type="file"] {
        width: 100%;
        min-height: 44px;
        border: 1px solid var(--line);
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.9);
        color: var(--ink);
        padding: 10px 12px;
      }
      button {
        min-height: 44px;
        border: 0;
        border-radius: 999px;
        background: linear-gradient(135deg, var(--accent), var(--accent-dark));
        color: white;
        font-weight: 850;
        cursor: pointer;
        box-shadow: 0 14px 30px rgba(11, 107, 91, 0.24);
        transition: transform 130ms ease, box-shadow 130ms ease;
      }
      button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 34px rgba(11, 107, 91, 0.30);
      }
      .secondary {
        background: var(--surface-strong);
        color: var(--ink);
        border: 1px solid var(--line);
        box-shadow: none;
      }
      .button-stack {
        display: grid;
        gap: 10px;
      }
      .button-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
      }
      .status {
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.62);
        color: var(--muted);
        padding: 12px;
        font-size: 0.91rem;
        line-height: 1.4;
      }
      .camera-panel, .prediction-panel {
        overflow: hidden;
        background: #0d1110;
      }
      .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 14px 16px;
        color: white;
      }
      .panel-header strong {
        font-size: 0.96rem;
      }
      .small-note {
        color: rgba(255, 255, 255, 0.62);
        font-size: 0.82rem;
      }
      .camera-frame, .prediction-frame {
        position: relative;
        aspect-ratio: 4 / 3;
        min-height: 420px;
        background: #111614;
      }
      .camera-frame video, .prediction-frame img {
        width: 100%;
        height: 100%;
        display: block;
        object-fit: contain;
      }
      .camera-frame video {
        object-fit: cover;
      }
      .frame-guide {
        position: absolute;
        inset: 18px;
        border: 1px solid rgba(255, 255, 255, 0.24);
        border-radius: 18px;
        pointer-events: none;
      }
      .empty-state {
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        padding: 24px;
        color: rgba(255, 255, 255, 0.66);
        text-align: center;
        line-height: 1.45;
      }
      .metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 0 16px 16px;
      }
      .metric {
        background: rgba(255, 255, 255, 0.10);
        border-color: rgba(255, 255, 255, 0.14);
        color: rgba(255, 255, 255, 0.78);
      }
      .toolbox {
        display: grid;
        grid-template-columns: minmax(260px, 0.7fr) minmax(360px, 1.3fr);
        gap: 16px;
        margin-top: 16px;
      }
      .upload-card, .upload-result {
        padding: 18px;
      }
      .upload-form {
        display: grid;
        gap: 12px;
        margin-top: 14px;
      }
      .upload-result-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
      }
      .upload-result img {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: #eef0ee;
      }
      .light-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
      }
      .light-metrics .metric {
        background: var(--accent-soft);
        border-color: transparent;
        color: var(--accent-dark);
      }
      .error {
        margin: 12px 0 0;
        color: #9a2424;
        font-weight: 800;
      }
      @media (max-width: 1180px) {
        .app-shell, .toolbox {
          grid-template-columns: 1fr;
        }
        .camera-frame, .prediction-frame {
          min-height: 320px;
        }
        .topbar {
          align-items: flex-start;
          flex-direction: column;
        }
        .badges {
          justify-content: flex-start;
        }
      }
      @media (max-width: 680px) {
        .page {
          width: min(100vw - 20px, 1480px);
          padding-top: 12px;
        }
        .upload-result-grid, .button-row {
          grid-template-columns: 1fr;
        }
        h1 {
          font-size: clamp(2rem, 18vw, 3.5rem);
        }
      }
    </style>
  </head>
  <body>
    <main class="page">
      <header class="topbar">
        <div>
          <p class="eyebrow">Deep learning hand tracking</p>
          <h1>Live hand landmark demo</h1>
        </div>
        <div>
          <p class="lead">
            Use the webcam as the primary demo, switch checkpoints quickly, then capture a frame or upload an image for controlled testing.
          </p>
          <div class="badges">
            <span class="badge">11 landmarks</span>
            <span class="badge">MediaPipe crop</span>
            <span class="badge">PyTorch backend</span>
          </div>
        </div>
      </header>

      <section class="app-shell">
        <aside class="panel controls">
          <div>
            <p class="eyebrow">Controls</p>
            <h2 class="panel-title">Run the model</h2>
            <p class="panel-copy">These settings are shared by live prediction, snapshot prediction, and image upload.</p>
          </div>
          <label>
            Model checkpoint
            <select name="checkpoint" id="checkpoint-select">
              {% for checkpoint_name in checkpoint_names %}
              <option value="{{ checkpoint_name }}" {% if checkpoint_name == selected_checkpoint %}selected{% endif %}>
                {{ checkpoint_name }}
              </option>
              {% endfor %}
            </select>
          </label>
          <label>
            Confidence threshold
            <select name="confidence_threshold" id="confidence-threshold">
              {% for value in ["0.10", "0.15", "0.20", "0.25", "0.30", "0.35"] %}
              <option value="{{ value }}" {% if value == selected_threshold %}selected{% endif %}>
                {{ value }}
              </option>
              {% endfor %}
            </select>
          </label>
          <div class="button-stack">
            <button type="button" id="start-camera">Start camera</button>
            <div class="button-row">
              <button type="button" class="secondary" id="capture-frame">Take picture</button>
              <button type="button" class="secondary" id="toggle-live">Start live</button>
            </div>
          </div>
          <div class="status" id="camera-status">Camera is idle. Start the camera to allow browser access.</div>
        </aside>

        <section class="panel camera-panel">
          <div class="panel-header">
            <strong>Webcam input</strong>
            <span class="small-note">local camera feed</span>
          </div>
          <div class="camera-frame">
            <video id="camera-preview" autoplay playsinline muted></video>
            <canvas id="camera-canvas" hidden></canvas>
            <div class="frame-guide"></div>
          </div>
        </section>

        <section class="panel prediction-panel" id="camera-results">
          <div class="panel-header">
            <strong>Live prediction</strong>
            <span class="small-note">model overlay</span>
          </div>
          <div class="prediction-frame">
            <img id="camera-prediction" src="{% if result_image %}data:image/jpeg;base64,{{ result_image }}{% endif %}" alt="Camera prediction" {% if not result_image %}style="display:none"{% endif %}>
            <div class="empty-state" id="camera-placeholder" {% if result_image %}style="display:none"{% endif %}>
              Start live prediction to see landmarks beside the camera feed.
            </div>
          </div>
          <div class="metrics">
            <span class="metric" id="camera-avg">avg confidence: {{ avg_confidence or "-" }}</span>
            <span class="metric" id="camera-min">min confidence: {{ min_confidence or "-" }}</span>
            <span class="metric" id="camera-max">max confidence: {{ max_confidence or "-" }}</span>
          </div>
        </section>
      </section>

      <section class="toolbox">
        <section class="panel upload-card">
          <p class="eyebrow">Still image test</p>
          <h2 class="panel-title">Upload an image</h2>
          <p class="panel-copy">Use this for report screenshots or quick checks on saved webcam frames.</p>
          <form method="post" enctype="multipart/form-data" class="upload-form" id="upload-form">
            <input type="hidden" name="checkpoint" id="upload-checkpoint" value="{{ selected_checkpoint }}">
            <input type="hidden" name="confidence_threshold" id="upload-threshold" value="{{ selected_threshold }}">
            <label>
              Image file
              <input type="file" name="image" accept=".jpg,.jpeg,.png,.bmp,.webp" required>
            </label>
            <button type="submit">Predict uploaded image</button>
          </form>
          {% if error %}
          <p class="error">{{ error }}</p>
          {% endif %}
        </section>

        <section class="panel upload-result">
          <p class="eyebrow">Latest result</p>
          {% if result_image %}
          <div class="upload-result-grid">
            <div>
              <strong>Input</strong>
              <img src="data:image/jpeg;base64,{{ original_image }}" alt="Original image">
            </div>
            <div>
              <strong>Prediction</strong>
              <img src="data:image/jpeg;base64,{{ result_image }}" alt="Prediction image">
            </div>
          </div>
          <div class="light-metrics">
            <span class="metric">avg confidence: {{ avg_confidence }}</span>
            <span class="metric">min confidence: {{ min_confidence }}</span>
            <span class="metric">max confidence: {{ max_confidence }}</span>
          </div>
          {% else %}
          <p class="panel-copy">No uploaded image result yet. Live output appears above, and upload results appear here.</p>
          {% endif %}
        </section>
      </section>
    </main>

    <script>
      const startButton = document.getElementById("start-camera");
      const captureButton = document.getElementById("capture-frame");
      const liveButton = document.getElementById("toggle-live");
      const video = document.getElementById("camera-preview");
      const canvas = document.getElementById("camera-canvas");
      const statusText = document.getElementById("camera-status");
      const checkpointSelect = document.getElementById("checkpoint-select");
      const thresholdSelect = document.getElementById("confidence-threshold");
      const cameraOriginal = document.createElement("img");
      const cameraPrediction = document.getElementById("camera-prediction");
      const cameraAvg = document.getElementById("camera-avg");
      const cameraMin = document.getElementById("camera-min");
      const cameraMax = document.getElementById("camera-max");
      const cameraPlaceholder = document.getElementById("camera-placeholder");
      const uploadForm = document.getElementById("upload-form");
      const uploadCheckpoint = document.getElementById("upload-checkpoint");
      const uploadThreshold = document.getElementById("upload-threshold");

      let mediaStream = null;
      let liveInterval = null;
      let isSending = false;

      async function ensureCamera() {
        if (mediaStream) {
          return true;
        }
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 960 }, height: { ideal: 720 } },
            audio: false
          });
          video.srcObject = mediaStream;
          statusText.textContent = "Camera ready. Start live prediction or take one picture.";
          return true;
        } catch (error) {
          statusText.textContent = "Could not access camera. Check browser permissions.";
          return false;
        }
      }

      function snapshotDataUrl() {
        const width = video.videoWidth;
        const height = video.videoHeight;
        if (!width || !height) {
          return null;
        }
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, width, height);
        return canvas.toDataURL("image/jpeg", 0.82);
      }

      async function runPrediction() {
        if (isSending) {
          return;
        }
        const ready = await ensureCamera();
        if (!ready) {
          return;
        }
        const imageData = snapshotDataUrl();
        if (!imageData) {
          statusText.textContent = "Camera is starting. Try again in a second.";
          return;
        }

        isSending = true;
        statusText.textContent = liveInterval ? "Live prediction running..." : "Running snapshot prediction...";

        try {
          const response = await fetch("/predict-api", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              image_data: imageData,
              checkpoint: checkpointSelect.value,
              confidence_threshold: thresholdSelect.value
            })
          });
          const payload = await response.json();
          if (!response.ok) {
            statusText.textContent = payload.error || "Prediction failed.";
            return;
          }

          cameraOriginal.src = payload.original_image;
          cameraPrediction.src = payload.result_image;
          cameraPrediction.style.display = "block";
          cameraPlaceholder.style.display = "none";
          cameraAvg.textContent = `avg confidence: ${payload.avg_confidence}`;
          cameraMin.textContent = `min confidence: ${payload.min_confidence}`;
          cameraMax.textContent = `max confidence: ${payload.max_confidence}`;
          statusText.textContent = liveInterval ? "Live prediction running." : "Snapshot prediction complete.";
        } catch (error) {
          statusText.textContent = "Prediction request failed.";
        } finally {
          isSending = false;
        }
      }

      startButton.addEventListener("click", async () => {
        await ensureCamera();
      });

      captureButton.addEventListener("click", async () => {
        await runPrediction();
      });

      liveButton.addEventListener("click", async () => {
        if (liveInterval) {
          clearInterval(liveInterval);
          liveInterval = null;
          liveButton.textContent = "Start live";
          statusText.textContent = "Live prediction stopped.";
          return;
        }

        const ready = await ensureCamera();
        if (!ready) {
          return;
        }

        liveButton.textContent = "Stop live";
        statusText.textContent = "Live prediction starting...";
        liveInterval = setInterval(runPrediction, 550);
        runPrediction();
      });

      uploadForm.addEventListener("submit", () => {
        uploadCheckpoint.value = checkpointSelect.value;
        uploadThreshold.value = thresholdSelect.value;
      });
    </script>
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
    parser.add_argument(
        "--disable-mediapipe",
        action="store_true",
        help="Skru av MediaPipe-handdeteksjon og bruk hele bildet direkte.",
    )
    parser.add_argument(
        "--mediapipe-padding",
        type=float,
        default=0.25,
        help="Ekstra padding rundt hand-boksen fra MediaPipe.",
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


def create_mediapipe_hands_detector():
    if mp is None:
        return None

    detector_kwargs = {
        "static_image_mode": False,
        "max_num_hands": 1,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }

    hands_factory = None
    solutions = getattr(mp, "solutions", None)
    if solutions is not None:
        hands_module = getattr(solutions, "hands", None)
        if hands_module is not None:
            hands_factory = getattr(hands_module, "Hands", None)

    if hands_factory is None:
        try:
            hands_module = importlib.import_module("mediapipe.python.solutions.hands")
            hands_factory = getattr(hands_module, "Hands", None)
        except Exception:
            hands_factory = None

    if hands_factory is None:
        print("MediaPipe is installed, but the Hands API is unavailable. Web app will use full-frame inference.")
        return None

    try:
        return hands_factory(**detector_kwargs)
    except Exception as exc:
        print(f"Failed to initialize MediaPipe Hands ({exc}). Web app will use full-frame inference.")
        return None


def compute_roi_from_mediapipe(
    image_bgr: np.ndarray,
    hands_detector,
    padding_ratio: float,
) -> tuple[int, int, int, int] | None:
    image_h, image_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None

    hand_landmarks = result.multi_hand_landmarks[0]
    xs = np.array([landmark.x * image_w for landmark in hand_landmarks.landmark], dtype=np.float32)
    ys = np.array([landmark.y * image_h for landmark in hand_landmarks.landmark], dtype=np.float32)

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    box_size = max(max_x - min_x, max_y - min_y)
    half_size = max(24.0, box_size * (0.5 + padding_ratio))

    x1 = max(0, int(np.floor(center_x - half_size)))
    y1 = max(0, int(np.floor(center_y - half_size)))
    x2 = min(image_w, int(np.ceil(center_x + half_size)))
    y2 = min(image_h, int(np.ceil(center_y + half_size)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


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


def run_inference_on_bgr_image(
    image_bgr: np.ndarray,
    checkpoint_path: Path,
    confidence_threshold: float,
    registry: CheckpointRegistry,
    device: torch.device,
    hands_detector=None,
    mediapipe_padding: float = 0.25,
) -> dict[str, str]:
    checkpoint, model, connections = registry.load(checkpoint_path)
    image_size = int(checkpoint.get("image_size", 224))

    roi = None
    image_for_model = image_bgr
    if hands_detector is not None:
        roi = compute_roi_from_mediapipe(
            image_bgr,
            hands_detector=hands_detector,
            padding_ratio=mediapipe_padding,
        )
        if roi is not None:
            x1, y1, x2, y2 = roi
            image_for_model = image_bgr[y1:y2, x1:x2]

    input_tensor = preprocess_image(image_for_model, image_size).to(device)
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
        target_hw=image_for_model.shape[:2],
    )
    if roi is not None:
        predicted_landmarks[:, 0] += x1
        predicted_landmarks[:, 1] += y1

    preview_bgr = draw_prediction(
        image_bgr,
        predicted_landmarks,
        connections,
        confidences,
        confidence_threshold=confidence_threshold,
    )
    if roi is not None:
        cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), (255, 210, 80), 2)

    return {
        "original_image": f"data:image/jpeg;base64,{encode_image(image_bgr)}",
        "result_image": f"data:image/jpeg;base64,{encode_image(preview_bgr)}",
        "avg_confidence": f"{float(confidences.mean()):.3f}",
        "min_confidence": f"{float(confidences.min()):.3f}",
        "max_confidence": f"{float(confidences.max()):.3f}",
    }


class CheckpointRegistry:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cache: dict[Path, tuple[dict[str, object], torch.nn.Module, list[tuple[int, int]]]] = {}

    def load(self, checkpoint_path: Path) -> tuple[dict[str, object], torch.nn.Module, list[tuple[int, int]]]:
        if checkpoint_path not in self.cache:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model = create_heatmap_model_from_checkpoint(checkpoint).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            selected_landmark_indices = checkpoint.get(
                "selected_landmark_indices",
                list(DEFAULT_LANDMARK_INDICES),
            )
            connections = infer_connections(selected_landmark_indices)
            self.cache[checkpoint_path] = (checkpoint, model, connections)
        return self.cache[checkpoint_path]


def create_app(
    default_checkpoint: Path,
    use_mediapipe: bool = True,
    mediapipe_padding: float = 0.25,
) -> Flask:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = available_checkpoints()
    checkpoint_map = {path.name: path for path in checkpoint_paths}
    registry = CheckpointRegistry(device)
    hands_detector = create_mediapipe_hands_detector() if use_mediapipe else None

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
                    confidence_threshold = float(selected_threshold)
                    result_payload = run_inference_on_bgr_image(
                        image_bgr=image_bgr,
                        checkpoint_path=checkpoint_path,
                        confidence_threshold=confidence_threshold,
                        registry=registry,
                        device=device,
                        hands_detector=hands_detector,
                        mediapipe_padding=mediapipe_padding,
                    )
                    original_image = result_payload["original_image"].split(",", 1)[1]
                    result_image = result_payload["result_image"].split(",", 1)[1]
                    avg_confidence = result_payload["avg_confidence"]
                    min_confidence = result_payload["min_confidence"]
                    max_confidence = result_payload["max_confidence"]

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

    @app.route("/predict-api", methods=["POST"])
    def predict_api():
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image_data")
        checkpoint_name = payload.get("checkpoint")
        threshold_value = payload.get("confidence_threshold", "0.25")

        if not image_data:
            return jsonify({"error": "Missing image data."}), 400
        if checkpoint_name not in checkpoint_map:
            return jsonify({"error": "Checkpoint does not exist."}), 400

        try:
            _, encoded_part = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded_part)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("Uploaded camera frame could not be decoded.")

            result_payload = run_inference_on_bgr_image(
                image_bgr=image_bgr,
                checkpoint_path=checkpoint_map[checkpoint_name],
                confidence_threshold=float(threshold_value),
                registry=registry,
                device=device,
                hands_detector=hands_detector,
                mediapipe_padding=mediapipe_padding,
            )
            return jsonify(result_payload)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def main() -> None:
    args = parse_args()
    app = create_app(
        args.default_checkpoint,
        use_mediapipe=not args.disable_mediapipe,
        mediapipe_padding=args.mediapipe_padding,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
