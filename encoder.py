import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms

# # === CONFIG ===
# MODEL_NAME = "microsoft/BiomedVLP-BioViL-T"
# TEXT_JSON_PATH = "metadata/processed/processed_test_metadata.json"
# IMAGE_NPY_DIR = "processed_data/test"
# SAVE_DIR = "embeddings/test"
# BATCH_SIZE = 32
# MAX_IMAGES = 2

# # === CONFIG ===
# MODEL_NAME = "microsoft/BiomedVLP-BioViL-T"
# TEXT_JSON_PATH = "metadata/processed/processed_valid_metadata.json"
# IMAGE_NPY_DIR = "processed_data/valid"
# SAVE_DIR = "embeddings/valid"
# BATCH_SIZE = 32
# MAX_IMAGES = 2

# === CONFIG ===
MODEL_NAME = "microsoft/BiomedVLP-BioViL-T"
TEXT_JSON_PATH = "metadata/processed/processed_train_metadata.json"
IMAGE_NPY_DIR = "processed_data/train"
SAVE_DIR = "embeddings/train"
BATCH_SIZE = 32
MAX_IMAGES = 2

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()

# Load metadata JSON
with open(TEXT_JSON_PATH, "r") as f:
    metadata = json.load(f)

def extract_study_uid(full_id: str) -> str:
    return full_id.split("_s")[-1]

# Prepare list of (StudyUID, text)
items = [
    (extract_study_uid(full_id), text)
    for full_id, text in metadata.items()
    if os.path.isfile(os.path.join(IMAGE_NPY_DIR, f"{extract_study_uid(full_id)}.npy"))
]

# Process in batches
for i in tqdm(range(0, len(items), BATCH_SIZE), desc="Generating Embeddings"):
    batch = items[i:i+BATCH_SIZE]
    study_uids = [uid for uid, _ in batch]
    texts = [text for _, text in batch]

    image_arrays = []
    valid_uids = []
    valid_texts = []

    for uid, text in zip(study_uids, texts):
        npy_path = os.path.join(IMAGE_NPY_DIR, f"{uid}.npy")
        try:
            arr = np.load(npy_path)  # shape: [N, 224, 224, 3]
            if arr.ndim != 4 or arr.shape[1:] != (224, 224, 3):
                print(f"Skipping {uid}: Invalid shape {arr.shape}")
                continue
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            continue

        arr = arr[-MAX_IMAGES:]
        N = arr.shape[0]

        if N < MAX_IMAGES:
            # Duplicate last image to fill missing slots
            num_to_duplicate = MAX_IMAGES - N
            duplicates = np.tile(arr[-1:], (num_to_duplicate, 1, 1, 1))
            arr = np.concatenate([arr, duplicates], axis=0)

        # Convert to [MAX_IMAGES, 3, 224, 224]
        arr = np.transpose(arr, (0, 3, 1, 2))
        image_arrays.append(arr)

        valid_uids.append(uid)
        valid_texts.append(text)

    if not image_arrays:
        print(f"Skipping batch {i // BATCH_SIZE}: No valid images")
        continue

    image_tensors = torch.tensor(np.stack(image_arrays)).float().to(device)

    # Normalize images (ImageNet stats)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_tensors = normalize(image_tensors)

    # Tokenize text
    encoded = tokenizer(
        valid_texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=image_tensors
        )

        # Use [CLS] token embedding as fused embedding
        last_hidden_state = outputs.last_hidden_state
        text_embeds = last_hidden_state[:, 0, :]  # shape: [batch_size, hidden_dim]

    # Save embeddings and metadata
    save_path = os.path.join(SAVE_DIR, f"batch_{i // BATCH_SIZE:04d}.pt")
    torch.save({
        "study_uids": valid_uids,
        "text_embeds": text_embeds.cpu()
    }, save_path)

    # Cleanup for memory
    del text_embeds, input_ids, attention_mask, image_tensors
    torch.cuda.empty_cache()

print(f"[INFO] All embeddings saved to: {SAVE_DIR}")
