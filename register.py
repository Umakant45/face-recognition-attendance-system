import cv2
import os
import sys

DATASET_PATH = "dataset"
CAPTURE_COUNT = 30  # Number of face samples to capture

def get_existing_ids(path):
    """Return a dict of {id: name} already registered."""
    users = {}
    if not os.path.exists(path):
        return users
    for file in os.listdir(path):
        if file.lower().endswith((".jpg", ".png")):
            parts = file.split("_")
            if len(parts) >= 2:
                try:
                    uid = int(parts[0])
                    name = parts[1]
                    users[uid] = name
                except ValueError:
                    pass
    return users

def register():
    os.makedirs(DATASET_PATH, exist_ok=True)

    existing = get_existing_ids(DATASET_PATH)

    # ── Get and validate User ID ──────────────────────────────────────────────
    while True:
        raw_id = input("Enter User ID (positive integer): ").strip()
        if not raw_id.isdigit() or int(raw_id) <= 0:
            print("  ✗ ID must be a positive integer. Try again.")
            continue
        user_id = int(raw_id)
        if user_id in existing:
            print(f"  ✗ ID {user_id} is already taken by '{existing[user_id]}'.")
            overwrite = input("  Overwrite existing data? (y/n): ").strip().lower()
            if overwrite != "y":
                continue
            # Remove old samples for this ID
            for f in os.listdir(DATASET_PATH):
                if f.startswith(f"{user_id}_"):
                    os.remove(os.path.join(DATASET_PATH, f))
            print(f"  Old data for ID {user_id} removed.")
        break

    # ── Get and validate User Name ────────────────────────────────────────────
    while True:
        user_name = input("Enter User Name (letters only, no spaces): ").strip()
        if not user_name.isalpha():
            print("  ✗ Name must contain letters only (no spaces or special characters).")
            continue
        break

    # ── Open camera ──────────────────────────────────────────────────────────
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("  ✗ Cannot open camera. Exiting.")
        sys.exit(1)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    print(f"\nRegistering '{user_name}' (ID: {user_id}).")
    print(f"Look at the camera. Capturing {CAPTURE_COUNT} face samples…")
    print("Press ESC to cancel at any time.\n")

    while True:
        ret, img = cam.read()
        if not ret:
            print("  ✗ Failed to read from camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            sample_path = os.path.join(DATASET_PATH, f"{user_id}_{user_name}_{count}.jpg")
            cv2.imwrite(sample_path, gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Progress overlay
        progress = f"Captured: {count}/{CAPTURE_COUNT}"
        cv2.putText(img, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow("Register Face — Press ESC to cancel", img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\nRegistration cancelled by user.")
            cam.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        if count >= CAPTURE_COUNT:
            break

    cam.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"\n✓ Face registration completed! {count} samples saved for '{user_name}'.")
        print("  Run train.py to update the model before taking attendance.")
    else:
        print("\n✗ No faces detected. Make sure your face is visible and well-lit.")

if __name__ == "__main__":
    register()
