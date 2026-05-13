import cv2
import os
import sys
import numpy as np
from PIL import Image, UnidentifiedImageError

DATASET_PATH = "dataset"
MODEL_PATH = "trainer.yml"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def load_training_data(path):
    faces = []
    ids = []
    skipped = []

    if not os.path.exists(path):
        print(f"✗ Dataset folder '{path}' not found. Register users first.")
        sys.exit(1)

    image_files = [
        f for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ]

    if not image_files:
        print(f"✗ No images found in '{path}'. Register users first.")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s). Loading…\n")

    for file_name in image_files:
        parts = file_name.split("_")
        if len(parts) < 2:
            skipped.append((file_name, "unexpected filename format"))
            continue

        try:
            user_id = int(parts[0])
        except ValueError:
            skipped.append((file_name, "ID is not an integer"))
            continue

        image_path = os.path.join(path, file_name)
        try:
            pil_img = Image.open(image_path).convert("L")
            img_array = np.array(pil_img, dtype="uint8")
            faces.append(img_array)
            ids.append(user_id)
        except (UnidentifiedImageError, OSError) as e:
            skipped.append((file_name, str(e)))

    return faces, ids, skipped

def train():
    faces, ids, skipped = load_training_data(DATASET_PATH)

    if skipped:
        print(f"⚠  Skipped {len(skipped)} file(s):")
        for name, reason in skipped:
            print(f"   • {name}: {reason}")
        print()

    if not faces:
        print("✗ No valid training data found. Exiting.")
        sys.exit(1)

    # Count samples per user for a summary
    from collections import Counter
    id_counts = Counter(ids)
    print("Training summary:")
    for uid, count in sorted(id_counts.items()):
        print(f"  ID {uid}: {count} sample(s)")
    print()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Training model…")
    recognizer.train(faces, np.array(ids))
    recognizer.save(MODEL_PATH)

    print(f"✓ Training completed! Model saved to '{MODEL_PATH}'.")
    print(f"  Total: {len(faces)} sample(s) across {len(id_counts)} user(s).")

if __name__ == "__main__":
    train()
