import os
import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, flash
)
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from gtts import gTTS
from difflib import SequenceMatcher
from googletrans import Translator

# 
#  Configuration
# 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # ← change if needed

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
IMAGE_FOLDER = "images"

ALLOWED_EXTENSIONS = {"mp4", "jpeg", "jpg", "png"}

# Languages you want to expose in the dropdown
LANGUAGE_CHOICES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "bn": "Bengali",
    "fr": "French",
}

# 
#  Flask setup
# 
app = Flask(__name__)
app.secret_key = "1234"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure folders exist
for path in (UPLOAD_FOLDER, STATIC_FOLDER, IMAGE_FOLDER):
    os.makedirs(path, exist_ok=True)

# 
#  Load super-resolution model (optional)
# 
sr = cv2.dnn_superres.DnnSuperResImpl_create()
model_path = "EDSR_x4.pb"
if os.path.exists(model_path):
    sr.readModel(model_path)
    sr.setModel("edsr", 2)
else:
    print(" Super-Resolution model not found continuing without it.")

#
#  Helper functions
# 
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def frame_capture(video_path: str):
    """Extract every 50-th frame and save to images/ directory."""
    vidcap = cv2.VideoCapture(video_path)
    count, idx, saved_frames = 0, 1, []

    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            break
        if count % 50 == 0:
            fpath = os.path.join(IMAGE_FOLDER, f"{idx}.jpg")
            cv2.imwrite(fpath, frame)
            saved_frames.append(fpath)
            idx += 1
        count += 1
    vidcap.release()
    return saved_frames


def preprocess_image(img_path: str):
    """Light denoise + adaptive thresholding for better OCR."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return img


def stitch_images(img_paths):
    """Simple OpenCV stitcher fallback to vertical concat."""
    imgs = [cv2.imread(p) for p in img_paths if cv2.imread(p) is not None]
    if not imgs:
        return None
    if len(imgs) == 1:
        return imgs[0]
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(imgs)
    return stitched if status == cv2.Stitcher_OK else np.vstack(imgs)



#  Routes
 
@app.route("/")
def index():
    return render_template("index.html", languages=LANGUAGE_CHOICES)


@app.route("/upload", methods=["POST"])
def upload():
    #  1. Validate file 
    file = request.files.get("file")
    if not file or file.filename == "" or not allowed_file(file.filename):
        flash("Please select a valid MP4 / image file.")
        return redirect(url_for("index"))

    out_lang = request.form.get("out_lang", "en").lower()
    if out_lang not in LANGUAGE_CHOICES:
        flash("Selected output language is not supported.")
        return redirect(url_for("index"))

    # 2. Save upload 
    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    #  3. Extract & stitch frames 
    frames = frame_capture(video_path)
    stitched = stitch_images(frames)
    if stitched is None:
        flash("Error processing video frames.")
        return redirect(url_for("index"))

    stitched_path = os.path.join(STATIC_FOLDER, "stitched.jpg")
    cv2.imwrite(stitched_path, stitched)

    #  4. Pre-process for OCR 
    processed = preprocess_image(stitched_path)
    processed_path = os.path.join(STATIC_FOLDER, "processed.jpg")
    cv2.imwrite(processed_path, processed)

    #  5. OCR (multi-language) 
    extracted_text = pytesseract.image_to_string(
        Image.open(processed_path).convert("RGB"),
        lang="eng+hin+tel+",
        config="--psm 6",
    ).strip()

    # Save raw OCR text for reference
    with open(os.path.join(STATIC_FOLDER, "ocr.txt"), "w", encoding="utf-8") as f:
        f.write(extracted_text)

    # 6. Translate to chosen language 
    translator = Translator()
    translated = translator.translate(extracted_text, dest=out_lang)
    translated_text = translated.text
    detected_lang = translated.src

    # 7. TTS in chosen language 
    audio_filename = f"tts_{out_lang}.mp3"
    audio_path = os.path.join(STATIC_FOLDER, audio_filename)
    gTTS(text=translated_text, lang=out_lang).save(audio_path)

    # 8. Optional accuracy (compare with ground_truth.txt) 
    accuracy = None
    gt_path = "ground_truth.txt"
    if os.path.exists(gt_path) and extracted_text:
        with open(gt_path, encoding="utf-8") as f:
            gt_text = f.read().strip()
        if gt_text:
            accuracy = SequenceMatcher(None, gt_text, extracted_text).ratio() * 100

    #9. Render result page 
    return render_template(
        "index1.html",
        file_content=extracted_text,
        translated_content=translated_text,
        language=LANGUAGE_CHOICES.get(detected_lang, detected_lang),
        output_language=LANGUAGE_CHOICES.get(out_lang, out_lang),
        audio_filename=audio_filename,
    )

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__== "__main__":
    app.run(host="127.0.0.1", port=3000, debug=True)