from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image
import io


# ============================================================
# AGRINEETI AI BACKEND
# ============================================================

app = Flask(__name__)

# Allow Angular frontend to communicate with Flask
CORS(app)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "agrineeti_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("AGRINEETI AI BACKEND")
print("=" * 60)

print("\nLoading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print(f"Supported classes: {len(class_names)}")

for class_name in class_names:
    print("-", class_name)


# ============================================================
# HOME API
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "AgriNeeti AI Backend is running",
        "status": "success",
        "classes": class_names
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "message": "No image uploaded."
            }), 400


        image_file = request.files["image"]

        if image_file.filename == "":

            return jsonify({
                "success": False,
                "message": "No image selected."
            }), 400


        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image_bytes = image_file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        # ----------------------------------------------------
        # RESIZE IMAGE
        # ----------------------------------------------------

        image = image.resize((224, 224))

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # IMPORTANT:
        # Do NOT apply preprocess_input here.
        # The trained AgriNeeti model already contains
        # the MobileNetV2 preprocessing layer.


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]


        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index] * 100
        )


        # ----------------------------------------------------
        # SPLIT CROP AND CONDITION
        # ----------------------------------------------------

        parts = predicted_class.split(
            "_",
            1
        )

        crop = parts[0]

        condition = (
            parts[1].replace("_", " ")
            if len(parts) > 1
            else "Unknown"
        )


        # ----------------------------------------------------
        # CONFIDENCE STATUS
        # ----------------------------------------------------

        if confidence < 50:

            confidence_level = "Low"

            message = (
                "The model is uncertain. "
                "Please upload another clear leaf image."
            )

        elif confidence < 75:

            confidence_level = "Moderate"

            message = (
                "Prediction completed. "
                "Consider another image for confirmation."
            )

        else:

            confidence_level = "High"

            message = "Prediction completed successfully."


        # ----------------------------------------------------
        # TOP 3 PREDICTIONS
        # ----------------------------------------------------

        top_indices = np.argsort(
            predictions
        )[-3:][::-1]

        top_predictions = []

        for index in top_indices:

            top_predictions.append({
                "class": class_names[int(index)],
                "confidence": round(
                    float(predictions[index] * 100),
                    2
                )
            })


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "crop": crop,

            "condition": condition,

            "confidence": round(
                confidence,
                2
            ),

            "confidenceLevel": confidence_level,

            "message": message,

            "topPredictions": top_predictions

        })


    except Exception as error:

        print("Prediction error:", error)

        return jsonify({
            "success": False,
            "message": "Prediction failed.",
            "error": str(error)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\nStarting AgriNeeti AI server...")
    print("API: http://127.0.0.1:5000")
    print("Prediction endpoint: http://127.0.0.1:5000/predict")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )