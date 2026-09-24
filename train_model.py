import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# AGRINEETI AI
# 4-CROP PLANT DISEASE CLASSIFICATION
#
# Crops:
# 1. Coconut
# 2. Cotton
# 3. Maize
# 4. Sugarcane
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Mithra BR\OneDrive\Desktop\AgriNeeti_AI"
)

DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 8

EPOCHS = 15

VALIDATION_SPLIT = 0.20

SEED = 42


print()
print("=" * 70)
print("AGRINEETI AI - 4 CROP PLANT DISEASE MODEL")
print("=" * 70)

print()
print("TensorFlow Version:", tf.__version__)
print("Dataset:", DATASET_DIR)
print()


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print("Loading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print()
print("Loading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

number_of_classes = len(class_names)

print()
print("=" * 70)
print("AGRINEETI MODEL CLASSES")
print("=" * 70)

for index, class_name in enumerate(class_names):
    print(
        f"{index:02d} -> {class_name}"
    )

print()
print(
    "Total Classes:",
    number_of_classes
)


# ============================================================
# VERIFY EXPECTED NUMBER OF CLASSES
# ============================================================

EXPECTED_CLASSES = 17

if number_of_classes != EXPECTED_CLASSES:
    print()
    print("WARNING")
    print(
        f"Expected {EXPECTED_CLASSES} classes, "
        f"but found {number_of_classes}."
    )
    print(
        "Please check the dataset folders before continuing."
    )

    raise ValueError(
        "Incorrect number of dataset classes."
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_file = (
    MODEL_DIR /
    "class_names.json"
)

with open(
    class_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


# ============================================================
# COUNT TRAINING IMAGES PER CLASS
# ============================================================

print()
print("=" * 70)
print("CLASS IMAGE COUNTS")
print("=" * 70)

class_counts = {}

total_images = 0

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

for class_name in class_names:

    class_folder = (
        DATASET_DIR /
        class_name
    )

    image_count = sum(
        1
        for file in class_folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in valid_extensions
    )

    class_counts[class_name] = image_count

    total_images += image_count

    print(
        f"{class_name:<35} : "
        f"{image_count}"
    )

print()
print(
    "Total Images:",
    total_images
)


# ============================================================
# CLASS WEIGHTS
#
# Dataset is highly imbalanced:
# Coconut has very few images.
# Sugarcane has 30 images/class.
# Cotton and Maize have hundreds/thousands.
#
# Class weights tell TensorFlow to give minority classes
# greater importance during training.
#
# We also cap extreme weights to keep training stable.
# ============================================================

print()
print("=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

class_weights = {}

for index, class_name in enumerate(class_names):

    count = class_counts[class_name]

    if count == 0:
        raise ValueError(
            f"No images found for {class_name}"
        )

    weight = (
        total_images /
        (
            number_of_classes *
            count
        )
    )

    # Avoid extremely large weights from the tiny
    # Coconut dataset destabilising training.
    weight = min(
        weight,
        10.0
    )

    class_weights[index] = weight

    print(
        f"{index:02d} "
        f"{class_name:<35} "
        f"Weight: {weight:.4f}"
    )


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = (
    train_dataset
    .prefetch(
        buffer_size=AUTOTUNE
    )
)

validation_dataset = (
    validation_dataset
    .prefetch(
        buffer_size=AUTOTUNE
    )
)


# ============================================================
# DATA AUGMENTATION
#
# Applied ONLY during training.
# Validation images are not augmented.
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.10
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        )
    ],
    name="data_augmentation"
)


# ============================================================
# MOBILENETV2 BASE MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING MOBILENETV2")
print("=" * 70)

base_model = MobileNetV2(
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)

# First training stage:
# Keep MobileNetV2 frozen.
base_model.trainable = False


# ============================================================
# CREATE MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    name="input_image"
)

x = data_augmentation(
    inputs
)

# IMPORTANT:
# Preprocessing stays inside the model.
# app.py must continue sending raw 0-255 image arrays.
x = preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(
    x
)

x = layers.Dropout(
    0.30,
    name="dropout"
)(
    x
)

outputs = layers.Dense(
    number_of_classes,
    activation="softmax",
    name="predictions"
)(
    x
)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="AgriNeeti_MobileNetV2"
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 70)
print("MODEL ARCHITECTURE")
print("=" * 70)

model.summary()


# ============================================================
# OUTPUT FILES
# ============================================================

model_path = (
    MODEL_DIR /
    "agrineeti_disease_model.keras"
)

best_model_path = (
    MODEL_DIR /
    "agrineeti_best_model.keras"
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("=" * 70)
print("STARTING 17-CLASS TRAINING")
print("=" * 70)

print()
print(
    "Crops: Coconut, Cotton, Maize, Sugarcane"
)

print(
    "Classes:",
    number_of_classes
)

print(
    "Images:",
    total_images
)

print(
    "Maximum epochs:",
    EPOCHS
)

print()
print(
    "Training with class weights to reduce "
    "the effect of dataset imbalance."
)

print()


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print()

print(
    f"Validation Loss     : "
    f"{loss:.4f}"
)

print(
    f"Validation Accuracy : "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    model_path
)


# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

training_info = {

    "crops": [
        "Coconut",
        "Cotton",
        "Maize",
        "Sugarcane"
    ],

    "numberOfClasses":
        number_of_classes,

    "totalImages":
        total_images,

    "imageSize": [
        IMAGE_SIZE[0],
        IMAGE_SIZE[1]
    ],

    "batchSize":
        BATCH_SIZE,

    "validationSplit":
        VALIDATION_SPLIT,

    "classes":
        class_names,

    "classCounts":
        class_counts,

    "classWeights": {
        class_names[index]:
            float(weight)

        for index, weight
        in class_weights.items()
    },

    "validationAccuracy":
        float(accuracy),

    "validationLoss":
        float(loss)
}


training_info_file = (
    MODEL_DIR /
    "training_info.json"
)

with open(
    training_info_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        training_info,
        file,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("AGRINEETI TRAINING COMPLETED")
print("=" * 70)

print()
print(
    "Final model:"
)

print(
    model_path
)

print()
print(
    "Best checkpoint:"
)

print(
    best_model_path
)

print()
print(
    "Class names:"
)

print(
    class_file
)

print()
print(
    "Training information:"
)

print(
    training_info_file
)

print()
print(
    f"Final Validation Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print()
print(
    "IMPORTANT:"
)

print(
    "This is a prototype plant-image classifier."
)

print(
    "Validation accuracy should not be interpreted "
    "as real-world diagnostic accuracy."
)

print()