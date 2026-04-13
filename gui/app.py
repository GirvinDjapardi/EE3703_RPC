"""
app.py — Flask server for the Retail Product Checkout Detector GUI.

Bonus deliverable for EE3703 Group 4.

Launch:
    cd gui/
    python3 app.py

Opens at http://localhost:5000
"""

import io
import base64
import sys
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from PIL import Image

# Ensure the gui/ package is importable
sys.path.insert(0, str(Path(__file__).parent))
from inference import (
    run_inference,
    YOLO_WEIGHTS,
    FASTERRCNN_WEIGHTS,
)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = True


@app.route("/")
def index():
    """Serve the main GUI page."""
    yolo_ready = YOLO_WEIGHTS.exists()
    rcnn_ready = FASTERRCNN_WEIGHTS.exists()
    return render_template(
        "index.html",
        yolo_ready=yolo_ready,
        rcnn_ready=rcnn_ready,
    )


@app.route("/detect", methods=["POST"])
def detect():
    """Run inference on the uploaded image.

    Expects multipart form data:
        image          : the uploaded image file
        model          : "YOLO" or "Faster R-CNN"
        conf_threshold : float between 0.0 and 1.0
    """
    # ── Validate inputs ──────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    model_name = request.form.get("model", "YOLO")
    if model_name not in ("YOLO", "Faster R-CNN"):
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    try:
        conf_threshold = float(request.form.get("conf_threshold", 0.25))
        conf_threshold = max(0.0, min(1.0, conf_threshold))
    except (ValueError, TypeError):
        conf_threshold = 0.25

    # ── Run inference ────────────────────────────────────────────
    image = Image.open(file.stream)
    annotated, detections, is_mock = run_inference(
        image, model_name, conf_threshold
    )

    # ── Encode annotated image as base64 JPEG ────────────────────
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # ── Build response ───────────────────────────────────────────
    return jsonify({
        "image": f"data:image/jpeg;base64,{img_b64}",
        "detections": [d.to_dict() for d in detections],
        "model": model_name,
        "is_mock": is_mock,
        "count": len(detections),
    })


if __name__ == "__main__":
    print("\n  Retail Checkout Detector GUI")
    print("  http://localhost:5050\n")
    app.run(debug=True, port=5050, use_reloader=False)
