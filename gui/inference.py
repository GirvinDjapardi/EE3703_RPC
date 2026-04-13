"""
inference.py — Unified inference wrapper for the Retail Checkout GUI.

Supports two backends:
  1. YOLO   (Ultralytics, weights: weights/yolo_best.pt)
  2. Faster R-CNN (PyTorch, weights: weights/fasterrcnn_best.pth)

When a weights file is missing the module falls back to **mock mode**,
which draws random bounding boxes so the GUI can be developed and
tested independently of the training pipeline.
"""

import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

# ── Project constants ────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Bottled Drink",
    1: "Canned Goods",
    2: "Fresh Produce",
    3: "Packaged Food",
}

# Distinct colour per class (RGB)
CLASS_COLOURS = {
    0: (14, 116, 144),    # teal       — Bottled Drink
    1: (220, 80, 50),     # warm red   — Canned Goods
    2: (245, 158, 11),    # amber      — Fresh Produce
    3: (34, 139, 34),     # forest grn — Packaged Food
}

WEIGHTS_DIR = Path(__file__).parent / "weights"
YOLO_WEIGHTS = WEIGHTS_DIR / "yolo best.pt"
FASTERRCNN_WEIGHTS = WEIGHTS_DIR / "frcnn best.pt"

IMG_SIZE = 640  # Expected input size (images should already be 640x640)

# Cached model instances (loaded once, reused across requests)
_yolo_model = None
_frcnn_model = None


# ── Detection result container ───────────────────────────────────────
class Detection:
    """One detected object."""

    def __init__(self, class_id: int, confidence: float,
                 x1: int, y1: int, x2: int, y2: int):
        self.class_id = class_id
        self.label = CLASS_NAMES.get(class_id, "Unknown")
        self.confidence = confidence
        self.box = (x1, y1, x2, y2)  # (left, top, right, bottom)

    def to_dict(self) -> dict:
        """Serialise for JSON response."""
        x1, y1, x2, y2 = self.box
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }


# ── Drawing constants (matched to panel_utils.py) ────────────────────
BOX_FILL_ALPHA = 64
BOX_BORDER_WIDTH = 4
LABEL_PADDING_X = 6
LABEL_PADDING_Y = 4
LABEL_FONT_SIZE = 16


def _load_label_font(size: int = LABEL_FONT_SIZE) -> ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _label_text_colour(fill: Tuple[int, int, int]) -> Tuple[int, int, int]:
    luminance = 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]
    return (0, 0, 0) if luminance >= 150 else (255, 255, 255)


def _rectangles_overlap(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
    padding: int = 4,
) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _clamp_badge_rect(
    x1: int, y1: int, w: int, h: int, cw: int, ch: int,
) -> Tuple[int, int, int, int]:
    cx = min(max(0, x1), max(0, cw - w - 1))
    cy = min(max(0, y1), max(0, ch - h - 1))
    return cx, cy, min(cw - 1, cx + w), min(ch - 1, cy + h)


def _choose_badge_rect(
    box: Tuple[int, int, int, int],
    bw: int, bh: int, cw: int, ch: int,
    occupied: List[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    for cx, cy in [
        (x1, y1 - bh - 4), (x2 - bw, y1 - bh - 4),
        (x1, y1 + 4),       (x2 - bw, y1 + 4),
        (x1, y2 - bh - 4), (x2 - bw, y2 - bh - 4),
        (x1, y2 + 4),       (x2 - bw, y2 + 4),
    ]:
        rect = _clamp_badge_rect(cx, cy, bw, bh, cw, ch)
        if all(not _rectangles_overlap(rect, o) for o in occupied):
            return rect
    for offset in range(0, ch, bh + 6):
        rect = _clamp_badge_rect(x1, y1 + offset, bw, bh, cw, ch)
        if all(not _rectangles_overlap(rect, o) for o in occupied):
            return rect
    return _clamp_badge_rect(x1, y1, bw, bh, cw, ch)


# ── Drawing utility ──────────────────────────────────────────────────
def draw_detections(image: Image.Image,
                    detections: List[Detection]) -> Image.Image:
    """Draw bounding boxes, class labels, and confidence scores on *image*.

    Uses the same RGBA overlay / rounded-badge style as panel_utils.py
    so the GUI output matches the notebook failure-analysis panels.
    """
    canvas = image.copy().convert("RGBA")
    font = _load_label_font()
    occupied: List[Tuple[int, int, int, int]] = []

    for det in detections:
        colour = CLASS_COLOURS.get(det.class_id, (255, 255, 255))
        x1, y1, x2, y2 = det.box

        overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        draw.rectangle([x1, y1, x2, y2], fill=colour + (BOX_FILL_ALPHA,))
        draw.rectangle([x1, y1, x2, y2], outline=colour + (255,), width=BOX_BORDER_WIDTH)

        label_text = f"{det.label} {det.confidence:.2f}"
        tb = draw.textbbox((0, 0), label_text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        bw = tw + LABEL_PADDING_X * 2
        bh = th + LABEL_PADDING_Y * 2

        bx1, by1, bx2, by2 = _choose_badge_rect(
            (x1, y1, x2, y2), bw, bh, canvas.width, canvas.height, occupied,
        )
        occupied.append((bx1, by1, bx2, by2))

        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=4, fill=colour + (230,))
        text_col = _label_text_colour(colour)
        draw.text(
            (bx1 + LABEL_PADDING_X, by1 + LABEL_PADDING_Y - tb[1]),
            label_text, fill=text_col, font=font,
        )

        canvas = Image.alpha_composite(canvas, overlay)

    return canvas.convert("RGB")


# ── Mock inference (used when weights are missing) ───────────────────
def _mock_inference(image: Image.Image,
                    conf_threshold: float) -> List[Detection]:
    """Return 2-5 random detections so the GUI can be tested."""
    w, h = image.size
    detections: List[Detection] = []
    for _ in range(random.randint(3, 6)):
        cls = random.randint(0, 3)
        conf = round(random.uniform(0.30, 0.98), 2)
        if conf < conf_threshold:
            continue
        # Generate boxes spread across the full image
        x1 = random.randint(10, int(w * 0.6))
        y1 = random.randint(10, int(h * 0.6))
        box_w = random.randint(80, min(220, w - x1 - 10))
        box_h = random.randint(80, min(220, h - y1 - 10))
        x2 = x1 + box_w
        y2 = y1 + box_h
        detections.append(Detection(cls, conf, x1, y1, x2, y2))
    return detections


# ── YOLO inference ───────────────────────────────────────────────────
def _yolo_inference(image: Image.Image,
                    conf_threshold: float) -> List[Detection]:
    """Run YOLO inference via Ultralytics."""
    global _yolo_model
    from ultralytics import YOLO

    if _yolo_model is None:
        _yolo_model = YOLO(str(YOLO_WEIGHTS))
    model = _yolo_model
    results = model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=conf_threshold,
        verbose=False,
    )

    detections: List[Detection] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append(Detection(cls, conf,
                                        int(x1), int(y1),
                                        int(x2), int(y2)))
    return detections


# ── Faster R-CNN inference ───────────────────────────────────────────
def _fasterrcnn_inference(image: Image.Image,
                          conf_threshold: float) -> List[Detection]:
    """Run Faster R-CNN inference using Naveen's from-scratch model."""
    global _frcnn_model
    import torch
    import torchvision.transforms.functional as TF
    from fasterrcnn_model import FasterRCNNFromScratch, IMAGE_MEAN, IMAGE_STD

    NUM_CLASSES = 4  # Bottled Drink, Canned Goods, Fresh Produce, Packaged Food

    if _frcnn_model is None:
        _frcnn_model = FasterRCNNFromScratch(NUM_CLASSES)
        checkpoint = torch.load(str(FASTERRCNN_WEIGHTS), map_location="cpu")
        _frcnn_model.load_state_dict(checkpoint["model_state"])
        _frcnn_model.eval()
    model = _frcnn_model

    # Preprocessing: to tensor then ImageNet normalisation
    img_tensor = TF.to_tensor(image)
    img_tensor = TF.normalize(img_tensor, IMAGE_MEAN, IMAGE_STD)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    detections: List[Detection] = []
    for box, label, score in zip(outputs["boxes"],
                                  outputs["labels"],
                                  outputs["scores"]):
        if float(score) < conf_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        # Model labels are 1-indexed (0 = background), GUI classes are 0-indexed
        cls = int(label) - 1
        detections.append(Detection(cls, float(score),
                                    int(x1), int(y1),
                                    int(x2), int(y2)))
    return detections


# ── Public entry point ───────────────────────────────────────────────
def run_inference(
    image: Image.Image,
    model_name: str,
    conf_threshold: float = 0.25,
) -> Tuple[Image.Image, List[Detection], bool]:
    """Run detection on *image* using *model_name*.

    Parameters
    ----------
    image : PIL.Image
        The uploaded image (expected to be 640x640 already).
    model_name : str
        Either ``"YOLO"`` or ``"Faster R-CNN"``.
    conf_threshold : float
        Minimum confidence to keep a detection (0.0–1.0).

    Returns
    -------
    annotated_image : PIL.Image
        The image with bounding boxes, labels, and confidence scores.
    detections : list[Detection]
        The raw detection objects for building the summary table.
    is_mock : bool
        True if mock inference was used (weights not found).
    """
    image = image.convert("RGB")

    # Choose backend
    is_mock = False
    if model_name == "YOLO":
        if YOLO_WEIGHTS.exists():
            detections = _yolo_inference(image, conf_threshold)
        else:
            detections = _mock_inference(image, conf_threshold)
            is_mock = True
    elif model_name == "Faster R-CNN":
        if FASTERRCNN_WEIGHTS.exists():
            detections = _fasterrcnn_inference(image, conf_threshold)
        else:
            detections = _mock_inference(image, conf_threshold)
            is_mock = True
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Draw results on the image
    annotated = draw_detections(image, detections)

    return annotated, detections, is_mock
