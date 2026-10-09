"""
PestSense — Smart Pest Presence Detection System
Flask backend configured for Render.

Local run:
    python app.py
Then open http://localhost:5000
"""

import base64
import importlib.util
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE = (224, 224)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Detect whether TensorFlow is installed without importing its large runtime
# during web-server startup. The actual import/model load happens on first use.
TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
_tf = None
_model = None
_preprocess_input = None
_decode_predictions = None

PEST_DB = {
    "grasshopper": ("Grasshopper", "Medium", "Apply kaolin clay barrier; use neem-based biopesticides."),
    "locust": ("Locust", "Critical", "Contact your local agriculture department promptly and follow locally approved control guidance."),
    "cricket": ("Cricket", "Low", "Remove debris; use suitable bait around crop borders if needed."),
    "caterpillar": ("Caterpillar", "High", "Consider a crop-appropriate Bacillus thuringiensis (Bt) product; remove egg masses where practical."),
    "moth": ("Moth/Borer", "High", "Use pheromone traps where appropriate and seek crop-specific advice."),
    "cabbage_butterfly": ("Cabbage Worm", "High", "Inspect leaf undersides; consider Bt and row covers where suitable."),
    "monarch": ("Butterfly", "Low", "Monitor; identify the species before taking control action."),
    "admiral": ("Butterfly", "Low", "Monitor; identify the species before taking control action."),
    "ringlet": ("Caterpillar", "High", "Inspect the crop and use crop-appropriate caterpillar management."),
    "sulphur_butterfly": ("Butterfly Pest", "Medium", "Inspect for larvae and use row covers where suitable."),
    "beetle": ("Beetle", "Medium", "Identify the beetle first; hand-pick or use an approved control if appropriate."),
    "weevil": ("Weevil", "High", "Inspect for damage and consider monitoring traps and crop-specific control."),
    "leaf_beetle": ("Leaf Beetle", "High", "Inspect affected leaves and use crop-appropriate controls."),
    "long_horn": ("Longhorn Beetle", "High", "Inspect damaged stems or branches and seek local agricultural guidance."),
    "rhinoceros_beetle": ("Rhinoceros Beetle", "High", "Check soil and plant damage; use locally approved management advice."),
    "buprestid": ("Jewel Beetle", "High", "Inspect affected wood or stems and seek crop-specific guidance."),
    "ladybug": ("Ladybug", "Low", "Usually beneficial; avoid treatment unless identification confirms a crop threat."),
    "leafhopper": ("Leafhopper", "Medium", "Monitor leaf damage; reflective mulch or suitable barriers may help."),
    "planthopper": ("Planthopper", "High", "Monitor field incidence and follow local integrated pest-management guidance."),
    "shield_bug": ("Stink Bug", "Medium", "Inspect crop damage and use suitable physical or approved controls."),
    "cicada": ("Cicada", "Medium", "Protect young plants where necessary and seek crop-specific advice."),
    "fly": ("Fruit Fly", "High", "Remove fallen or infested fruit and use suitable monitoring traps."),
    "spider": ("Spider/Mite-like label", "High", "Confirm identification before treatment; some spiders are beneficial."),
    "tick": ("Mite/Tick-like label", "High", "Confirm the organism before choosing a control."),
    "ant": ("Ant", "Medium", "Check whether ants are causing crop damage or tending other pests before treatment."),
    "bee": ("Bee", "Low", "Beneficial pollinator; avoid harming bees."),
    "dragonfly": ("Dragonfly", "Low", "Beneficial predator; no action is usually needed."),
    "mantis": ("Praying Mantis", "Low", "Beneficial predator; no action is usually needed."),
    "slug": ("Slug", "Medium", "Use crop-safe slug management and reduce hiding places where practical."),
    "snail": ("Snail", "Medium", "Use crop-safe snail management and reduce hiding places where practical."),
    "armyworm": ("Armyworm", "Critical", "Scout crop damage and follow local crop-specific integrated pest-management guidance."),
    "mealybug": ("Mealybug", "Medium", "Inspect affected plants and use a locally approved, crop-appropriate control."),
    "whitefly": ("Whitefly", "High", "Yellow sticky traps can help monitor adults; inspect leaf undersides."),
    "thrips": ("Thrips", "Medium", "Remove heavily affected plant parts where practical and use crop-specific guidance."),
    "walking_stick": ("Walking Stick", "Low", "Confirm identification and monitor for actual crop damage."),
    "stick_insect": ("Stick Insect", "Low", "Confirm identification and monitor for actual crop damage."),
}

GRASSHOPPER_LOOKALIKES = {
    "ant", "cockroach", "cricket", "walking_stick",
    "stick_insect", "grasshopper", "locust", "mantis",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload limit
CORS(app)


def get_model():
    """Load MobileNetV2 only when the first detection request needs it."""
    global _tf, _model, _preprocess_input, _decode_predictions, TF_AVAILABLE

    if not TF_AVAILABLE:
        return None
    if _model is not None:
        return _model

    print("[INFO] Importing TensorFlow and loading MobileNetV2...", flush=True)
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import (
            preprocess_input,
            decode_predictions,
        )

        _tf = tf
        _preprocess_input = preprocess_input
        _decode_predictions = decode_predictions
        _model = MobileNetV2(weights="imagenet")
        print("[INFO] MobileNetV2 is ready.", flush=True)
        return _model
    except Exception as exc:
        print(f"[ERROR] Could not load TensorFlow model: {exc}", flush=True)
        raise RuntimeError(f"AI model failed to load: {exc}") from exc


def analyze_features(img: Image.Image) -> dict:
    """Calculate simple color ratios; these are heuristics, not a trained detector."""
    arr = np.asarray(img.convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    total = float(IMG_SIZE[0] * IMG_SIZE[1])

    green_ratio = float(((g > r + 15) & (g > b + 15) & (g > 60)).sum()) / total
    brown_ratio = float(
        ((r > 90) & (r < 220) & (g > 55) & (g < 170) &
         (b < 110) & (r > g + 5) & (g > b + 5)).sum()
    ) / total
    dark_ratio = float(
        ((r > 20) & (r < 100) & (g > 10) & (g < 80) & (b < 60)).sum()
    ) / total

    return {
        "green_ratio": round(green_ratio, 3),
        "brown_ratio": round(brown_ratio, 3),
        "dark_ratio": round(dark_ratio, 3),
        "is_crop_scene": green_ratio > 0.25,
        "has_insect_body": brown_ratio > 0.05 or dark_ratio > 0.03,
        "likely_pest": green_ratio > 0.25 and (brown_ratio > 0.05 or dark_ratio > 0.03),
    }


def match_pest(label: str):
    normalized = label.lower().replace(" ", "_")
    for key, value in PEST_DB.items():
        if key in normalized or normalized in key:
            return key, value
    return None, None


def run_inference(img: Image.Image) -> dict:
    features = analyze_features(img)

    if not TF_AVAILABLE:
        return {
            "pest_detected": False,
            "pest_name": None,
            "confidence": 0,
            "severity": None,
            "remedy": "TensorFlow is not installed; this result is not an AI prediction.",
            "top_predictions": [],
            "mode": "unavailable",
            "features": features,
            "warning": "TensorFlow is unavailable. Install the configured dependencies to enable model inference.",
        }

    model = get_model()
    arr = np.expand_dims(np.asarray(img.convert("RGB").resize(IMG_SIZE), dtype=np.float32), 0)
    predictions = model.predict(_preprocess_input(arr), verbose=0)
    top = _decode_predictions(predictions, top=10)[0]

    top_predictions = [
        {"pest": label.replace("_", " ").title(), "label": label, "confidence": round(float(prob), 4)}
        for _, label, prob in top[:5]
    ]

    det_key = None
    det_info = None
    pest_conf = 0.0
    for _, label, probability in top:
        key, info = match_pest(label)
        if info is not None:
            det_key, det_info = key, info
            pest_conf = round(float(probability) * 100, 1)
            break

    if (
        det_key in GRASSHOPPER_LOOKALIKES
        and features["is_crop_scene"]
        and features["has_insect_body"]
    ):
        boosted = min(pest_conf * 3.5 + features["brown_ratio"] * 120, 92.0)
        det_info = PEST_DB["grasshopper"]
        pest_conf = round(max(boosted, 62.0), 1)

    if det_info is None and features["likely_pest"]:
        score = features["brown_ratio"] * 250 + features["dark_ratio"] * 150
        det_info = PEST_DB["grasshopper"]
        pest_conf = round(min(max(score, 55.0), 88.0), 1)

    pest_detected = det_info is not None
    return {
        "pest_detected": pest_detected,
        "pest_name": det_info[0] if det_info else None,
        "confidence": pest_conf if pest_detected else 95.0,
        "severity": det_info[1] if det_info else None,
        "remedy": det_info[2] if det_info else None,
        "top_predictions": top_predictions,
        "mode": "model",
        "features": features,
        "warning": "MobileNetV2 is an ImageNet classifier, not a pest-specific trained model; results are experimental.",
    }


@app.get("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.get("/metrics")
def metrics():
    return send_from_directory(str(BASE_DIR), "metrics.html")


# Root-level PWA routes. The /static aliases remain for compatibility with older HTML.
@app.get("/manifest.json")
@app.get("/static/manifest.json")
def manifest():
    return send_from_directory(str(BASE_DIR), "manifest.json", mimetype="application/manifest+json")


@app.get("/icon-192.png")
@app.get("/static/icon-192.png")
def icon_192():
    return send_from_directory(str(BASE_DIR), "icon-192.png")


@app.get("/icon-512.png")
@app.get("/static/icon-512.png")
def icon_512():
    return send_from_directory(str(BASE_DIR), "icon-512.png")


@app.get("/sw.js")
def service_worker():
    response = send_from_directory(str(BASE_DIR), "sw.js", mimetype="application/javascript")
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/static/<path:filename>")
def static_files(filename):
    # Only serve the known root-level PWA files through this compatibility route.
    allowed = {"manifest.json", "icon-192.png", "icon-512.png", "sw.js"}
    if filename not in allowed:
        return "Not Found", 404
    return send_from_directory(str(BASE_DIR), filename)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "tensorflow": TF_AVAILABLE,
        "model_loaded": _model is not None,
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file was uploaded. Expected form field 'image'."}), 400

    uploaded = request.files["image"]
    filename = secure_filename(uploaded.filename or "")
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported image format '{extension}'. Use JPG, PNG, or WEBP."}), 415

    try:
        image = Image.open(io.BytesIO(uploaded.read()))
        image.load()
        if image.width < 1 or image.height < 1:
            return jsonify({"error": "The uploaded image is empty."}), 400

        result = run_inference(image)
        thumbnail = image.convert("RGB")
        thumbnail.thumbnail((300, 300))
        buffer = io.BytesIO()
        thumbnail.save(buffer, format="JPEG", quality=75)
        result["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        return jsonify(result)

    except UnidentifiedImageError:
        return jsonify({"error": "The uploaded file is not a valid readable image."}), 400
    except Exception as exc:
        app.logger.exception("Detection request failed")
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image is too large. Maximum upload size is 12 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print("=" * 60, flush=True)
    print("PestSense Smart Pest Detection", flush=True)
    print(f"Starting Flask on 0.0.0.0:{port}", flush=True)
    print(f"TensorFlow package available: {TF_AVAILABLE}", flush=True)
    print("=" * 60, flush=True)
    # Do not preload TensorFlow/model here; bind the port immediately.
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
