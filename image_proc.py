import cv2
import os
import numpy as np
from collections import defaultdict
import pathlib
from tqdm import tqdm
import signal
import sys

# --- Configuration ---
image_dir = 'data/deid_png'
output_dir = 'normalised_processed_data'
target_size = (224, 224)
padding_value = 0  # Consider non-zero padding if zeros cause issues

# --- Setup ---
pathlib.Path(output_dir).mkdir(exist_ok=True)
interrupted = False

# --- Handle keyboard interrupt ---
def handle_interrupt(sig, frame):
    global interrupted
    interrupted = True
    print("\n🛑 KeyboardInterrupt received. Finishing current study and exiting...")

signal.signal(signal.SIGINT, handle_interrupt)

# --- Group image paths by study ID ---
study_images = defaultdict(list)
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.png'):
            full_path = os.path.join(root, file)
            parts = full_path.split(os.sep)
            if 'studies' in parts:
                studies_idx = parts.index('studies')
                study_id = parts[studies_idx + 1]
                study_images[study_id].append(full_path)

for study_id in study_images:
    study_images[study_id].sort()

# --- Skip already processed studies ---
processed_ids = set(os.listdir(output_dir))
processed_ids = {d for d in processed_ids if os.path.isdir(os.path.join(output_dir, d))}

# --- Resize with aspect ratio ---
def resize_with_aspect_ratio(img, target_size=(224, 224), padding_value=0):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    pad_w, pad_h = target_w - new_w, target_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    if len(resized.shape) == 2:
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)
    else:
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[padding_value]*3)
    return padded

# --- Main loop ---
total_images = 0
study_ids = list(study_images.keys())

print(f"📁 Found {len(study_ids)} studies.")

for study_id in tqdm(study_ids, desc="Processing studies"):
    if study_id in processed_ids:
        continue

    processed_images = []

    for img_path in study_images[study_id]:
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"⚠️ Warning: Failed to load {img_path}")
            continue

        # Handle bit depth
        if img.dtype == np.uint8:
            img = img.astype(np.float32) * (65535 / 255)  # Scale to 16-bit range
            max_value = 65535
        elif img.dtype == np.uint16:
            max_value = 65535
        else:
            print(f"⚠️ Warning: Unexpected dtype {img.dtype} for {img_path}")
            continue

        # Clip to valid range
        img = np.clip(img, 0, max_value)

        # Invert (common for CXRs)
        img_inverted = max_value - img
        img_inverted = np.clip(img_inverted, 0, max_value)

        # Normalize to [0,1]
        img_norm = img_inverted.astype(np.float32) / max_value

        # Convert to grayscale if needed
        if len(img_norm.shape) > 2:
            img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)

        # Check for invalid data
        if np.any(np.isnan(img_norm)) or np.all(img_norm == 0):
            print(f"⚠️ Warning: Invalid data (NaN or all zeros) in {img_path}")
            continue

        # Resize with aspect ratio
        img_resized = resize_with_aspect_ratio(img_norm, target_size=target_size, padding_value=padding_value)

        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_resized = clahe.apply((img_resized * 255).astype(np.uint8)).astype(np.float32) / 255

        # Final clip to ensure [0,1]
        img_resized = np.clip(img_resized, 0, 1)

        # Stack to 3 channels for BioViL-T
        img_resized = np.stack([img_resized] * 3, axis=-1)

        processed_images.append(img_resized)

    if processed_images:
        stacked = np.stack(processed_images, axis=0)  # Shape: (num_images, 224, 224, 3)
        output_study_dir = os.path.join(output_dir, study_id)
        pathlib.Path(output_study_dir).mkdir(exist_ok=True)
        output_path = os.path.join(output_study_dir, f"{study_id}.npy")
        np.save(output_path, stacked)
        total_images += len(processed_images)
    else:
        print(f"⚠️ No valid images found in study {study_id}")

    if interrupted:
        print(f"\n🛑 Interrupted. Saved progress up to study {study_id}.")
        break

# --- Final report ---
print("\n✅ Preprocessing complete.")
print(f"📁 Total studies processed: {len(study_ids) - len(processed_ids)}")
print(f"🖼️ Total images preprocessed: {total_images}")