import flask
from flask import Flask, request, jsonify
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

model_path = "/Users/shrish/Desktop/AIML Assignment/model/runs/detect/train2/weights/best.pt"
model = YOLO(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file)

    # Run inference
    results = model(img)

    # Convert results to JSON
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": int(box.cls),
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            })

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(debug=True)

