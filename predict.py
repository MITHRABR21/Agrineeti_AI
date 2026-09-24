import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# AGRINEETI AI - PLANT DISEASE PREDICTION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Mithra BR\OneDrive\Desktop\AgriNeeti_AI"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "agrineeti_disease_model.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_DIR
    / "models"
    / "class_names.json"
)

TEST_IMAGES_DIR = (
    PROJECT_DIR
    / "test_images"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("AGRINEETI AI")
print("PLANT DISEASE DETECTION")
print("=" * 60)

print()
print("Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)


print()
print(
    f"Supported classes: "
    f"{len(class_names)}"
)


# ============================================================
# FIND TEST IMAGE
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


test_images = [
    file
    for file in TEST_IMAGES_DIR.iterdir()
    if (
        file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    )
]


if len(test_images) == 0:

    print()
    print(
        "ERROR: No test image found."
    )

    print(
        "Put one Coconut or Sugarcane "
        "image inside:"
    )

    print(
        TEST_IMAGES_DIR
    )

    sys.exit()


# Use first image found
image_path = test_images[0]


print()
print(
    "Testing image:"
)

print(
    image_path.name
)


# ============================================================
# PREPARE IMAGE
# ============================================================

image = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE
)

image_array = (
    tf.keras.utils.img_to_array(
        image
    )
)

image_array = tf.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("Analyzing leaf image...")

predictions = model.predict(
    image_array,
    verbose=0
)

scores = predictions[0]


predicted_index = int(
    np.argmax(scores)
)

predicted_class = (
    class_names[predicted_index]
)

confidence = float(
    scores[predicted_index]
) * 100


# ============================================================
# FORMAT RESULT
# ============================================================

parts = predicted_class.split(
    "_",
    1
)

crop = parts[0]

condition = (
    parts[1]
    if len(parts) > 1
    else predicted_class
)

condition = condition.replace(
    "_",
    " "
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 60)
print("AGRINEETI PREDICTION RESULT")
print("=" * 60)

print()
print(
    f"Crop       : {crop}"
)

print(
    f"Condition  : {condition}"
)

print(
    f"Confidence : {confidence:.2f}%"
)


# ============================================================
# TOP 3 PREDICTIONS
# ============================================================

top_indices = np.argsort(
    scores
)[-3:][::-1]


print()
print("Top 3 Predictions:")
print("-" * 60)


for rank, index in enumerate(
    top_indices,
    start=1
):

    class_name = (
        class_names[int(index)]
    )

    probability = (
        float(scores[index])
        * 100
    )

    print(
        f"{rank}. "
        f"{class_name} "
        f"- {probability:.2f}%"
    )


print()
print("=" * 60)
print("Prediction completed.")
print("=" * 60)