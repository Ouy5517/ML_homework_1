"""Flask API for the RideCast demand estimator."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

try:
    from .model import frame_from_payload, load_pipeline
except ImportError:  # pragma: no cover - supports direct script execution
    from model import frame_from_payload, load_pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "ridecast_pipeline.joblib"


def create_app(model=None) -> Flask:
    app = Flask(__name__, static_folder=str(ROOT), static_url_path="")
    configured_model_path = Path(os.getenv("RIDECAST_MODEL_PATH", DEFAULT_MODEL_PATH))
    active_model = model
    if active_model is None and configured_model_path.exists():
        active_model = load_pipeline(configured_model_path)

    @app.get("/")
    def index():
        return send_from_directory(ROOT, "index.html")

    @app.get("/health")
    def health():
        loaded = active_model is not None
        return jsonify({"status": "ok" if loaded else "degraded", "model_loaded": loaded}), (200 if loaded else 503)

    @app.post("/predict")
    def predict():
        if active_model is None:
            return jsonify({"error": "模型文件不存在，请先运行 python backend/train_model.py"}), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "请求体必须是 JSON 对象"}), 400
        try:
            features = frame_from_payload(payload)
            value = float(active_model.predict(features)[0])
        except (ValueError, TypeError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"prediction": max(0, round(value))})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
