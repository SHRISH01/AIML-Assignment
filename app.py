from flask import Flask, request, render_template, jsonify
import torch
from ultralytics import YOLO
import os
from PIL import Image, ImageFile

app = Flask(__name__)

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_PATH = "model/runs/detect/train2/weights/best.pt"
model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Save uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        try:
            # Open the image
            img = Image.open(image_path)
            results = model(img)

            # Convert results to JSON
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        "class": int(box.cls),
                        "confidence": float(box.conf),
                        "bbox": [float(coord) for coord in box.xyxy[0]]
                    })

            return render_template("index.html", uploaded_image=image_path, detections=detections)
        
        except OSError:
            return jsonify({"error": "Corrupted or unsupported image file"}), 400

    return render_template("index.html", uploaded_image=None, detections=None)

if __name__ == "__main__":
    app.run(debug=True)
