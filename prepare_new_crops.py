from pathlib import Path
import shutil

# ============================================================
# AGRINEETI - PREPARE MAIZE + COTTON DATASETS
# ============================================================

MAIZE_SOURCE = Path(
    r"C:\Users\Mithra BR\Downloads\archive (4)\data"
)

COTTON_SOURCE = Path(
    r"C:\Users\Mithra BR\Downloads\archive (3)\cotton"
)

TARGET_DATASET = Path(
    r"C:\Users\Mithra BR\OneDrive\Desktop\AgriNeeti_AI\dataset"
)

# Source folder name -> AgriNeeti folder name
MAIZE_CLASSES = {
    "Blight": "Maize_Blight",
    "Common_Rust": "Maize_Common_Rust",
    "Gray_Leaf_Spot": "Maize_Gray_Leaf_Spot",
    "Healthy": "Maize_Healthy",
}

COTTON_CLASSES = {
    "bacterial_blight": "Cotton_Bacterial_Blight",
    "curl_virus": "Cotton_Curl_Virus",
    "fussarium_wilt": "Cotton_Fusarium_Wilt",
    "healthy": "Cotton_Healthy",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_class(source_folder, target_folder):
    if not source_folder.exists():
        print(f"ERROR: Source folder not found: {source_folder}")
        return 0

    target_folder.mkdir(parents=True, exist_ok=True)

    count = 0

    for file in source_folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            destination = target_folder / file.name

            # Prevent accidental overwrite if duplicate filenames exist
            if destination.exists():
                stem = file.stem
                suffix = file.suffix

                duplicate_number = 1

                while destination.exists():
                    destination = target_folder / (
                        f"{stem}_{duplicate_number}{suffix}"
                    )
                    duplicate_number += 1

            shutil.copy2(file, destination)
            count += 1

    return count


print("=" * 65)
print("AGRINEETI - PREPARING NEW CROP DATASETS")
print("=" * 65)

print("\nTarget dataset:")
print(TARGET_DATASET)

TARGET_DATASET.mkdir(parents=True, exist_ok=True)

# ============================================================
# MAIZE
# ============================================================

print("\n" + "=" * 65)
print("PREPARING MAIZE")
print("=" * 65)

for source_name, target_name in MAIZE_CLASSES.items():

    source = MAIZE_SOURCE / source_name
    target = TARGET_DATASET / target_name

    print(f"\n{source_name} -> {target_name}")

    count = copy_class(source, target)

    print(f"Copied: {count} images")


# ============================================================
# COTTON
# ============================================================

print("\n" + "=" * 65)
print("PREPARING COTTON")
print("=" * 65)

for source_name, target_name in COTTON_CLASSES.items():

    source = COTTON_SOURCE / source_name
    target = TARGET_DATASET / target_name

    print(f"\n{source_name} -> {target_name}")

    count = copy_class(source, target)

    print(f"Copied: {count} images")


# ============================================================
# FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL AGRINEETI DATASET")
print("=" * 65)

total_images = 0
total_classes = 0

for folder in sorted(TARGET_DATASET.iterdir()):

    if folder.is_dir():

        image_count = sum(
            1
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{folder.name:<35} : {image_count}")

        total_images += image_count
        total_classes += 1


print("\n" + "=" * 65)
print(f"Total Classes : {total_classes}")
print(f"Total Images  : {total_images}")
print("=" * 65)

print("\nDataset preparation completed successfully.")
print("Do NOT train the model yet.")