"""
app.py - Flask backend for the Alpha / Beta captioning demo.

Place this file next to load_alpha_beta_models.py (same folder that holds
the alpha/, beta/ and images/ directories), then run:

    pip install flask
    python app.py

Open http://127.0.0.1:5000
"""

import io
import os
import threading
import time
from pathlib import Path

import torch
from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image, ImageOps
from timm.data import create_transform, resolve_model_data_config

# Checkpoint paths in load_alpha_beta_models.py are relative, so make sure
# they resolve from this folder no matter where the server is launched.
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

import loading_testing_normalized as lab  # noqa: E402

IMAGES_DIR = BASE_DIR / os.environ.get("IMAGES_DIR", "images")
STATIC_DIR = BASE_DIR / "static"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB uploads

MODELS = {}          # {"alpha": (model, tokenizer), "beta": (model, tokenizer)}
BETA_TRANSFORM = None
INFER_LOCK = threading.Lock()  # one forward pass at a time (GPU / CPU safety)


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_models():
    global BETA_TRANSFORM
    print(f"Using device: {lab.DEVICE}\n")

    print("Loading Blueprint Alpha ...")
    MODELS["alpha"] = lab.load_alpha_model(lab.ALPHA_CHECKPOINT_PATH, device=lab.DEVICE)
    print("Alpha ready.\n")

    print("Loading Blueprint Beta ...")
    MODELS["beta"] = lab.load_beta_model(lab.BETA_CHECKPOINT_PATH, device=lab.DEVICE)
    print("Beta ready.\n")

    beta_model, _ = MODELS["beta"]
    cfg = resolve_model_data_config(beta_model.vision_encoder)
    BETA_TRANSFORM = create_transform(**cfg, is_training=False)


# ----------------------------------------------------------------------
# Inference (mirrors test_models_on_image from the loader script)
# ----------------------------------------------------------------------

def run_alpha(image: Image.Image) -> str:
    model, tokenizer = MODELS["alpha"]
    pixel_values = lab.letterbox_pil(image, size=512).unsqueeze(0).to(lab.DEVICE)

    tokens = tokenizer("Describe this image.", return_tensors="pt", padding=True)
    input_ids = tokens["input_ids"].to(lab.DEVICE)
    attention_mask = tokens["attention_mask"].to(lab.DEVICE)

    with torch.no_grad():
        output_ids = model.generate_caption(
            pixel_values, input_ids, attention_mask,
            max_new_tokens=32, num_beams=4,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def run_beta(image: Image.Image) -> str:
    model, _ = MODELS["beta"]
    pixel_values = BETA_TRANSFORM(image).unsqueeze(0).to(lab.DEVICE)
    prompt = "Describe this image in one sentence:"

    with torch.no_grad():
        captions = model.generate_caption(pixel_values, prompt, max_new_tokens=30)
    return captions[0].strip()


RUNNERS = {"alpha": run_alpha, "beta": run_beta}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def open_rgb(stream_or_path) -> Image.Image:
    image = Image.open(stream_or_path)
    image = ImageOps.exif_transpose(image)  # respect phone-camera rotation
    return image.convert("RGB")


def safe_image_path(name: str) -> Path:
    root = IMAGES_DIR.resolve()
    path = (root / name).resolve()
    if root not in path.parents or path.suffix.lower() not in ALLOWED_EXT or not path.is_file():
        abort(404)
    return path


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify(device=str(lab.DEVICE), ready=sorted(MODELS.keys()))


@app.get("/api/images")
def list_images():
    if not IMAGES_DIR.is_dir():
        return jsonify(images=[])
    files = sorted(
        p.name for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT
    )
    return jsonify(images=[{"name": n, "url": f"/images/{n}"} for n in files])


@app.get("/images/<path:name>")
def serve_image(name):
    path = safe_image_path(name)
    return send_from_directory(path.parent, path.name)


@app.post("/api/caption/<model_key>")
def caption(model_key):
    if model_key not in RUNNERS:
        return jsonify(error=f"Unknown model '{model_key}'."), 404

    try:
        if "file" in request.files:
            upload = request.files["file"]
            image = open_rgb(io.BytesIO(upload.read()))
        else:
            payload = request.get_json(silent=True) or {}
            name = payload.get("name")
            if not name:
                return jsonify(error="Send an image file or the name of a gallery image."), 400
            image = open_rgb(safe_image_path(name))
    except Exception as exc:  # unreadable / unsupported image
        return jsonify(error=f"Could not read the image: {exc}"), 400

    try:
        with INFER_LOCK:
            start = time.perf_counter()
            text = RUNNERS[model_key](image)
            if lab.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = int((time.perf_counter() - start) * 1000)
    except Exception as exc:
        app.logger.exception("Inference failed for %s", model_key)
        return jsonify(error=f"{model_key} failed: {exc}"), 500

    return jsonify(model=model_key, caption=text, ms=elapsed_ms)


if __name__ == "__main__":
    load_models()
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False, use_reloader=False)