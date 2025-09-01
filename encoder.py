import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms

# === CONFIG ===
MODEL_NAME = "microsoft/BiomedVLP-BioViL-T"
TEXT_JSON_PATH = "metadata/processed/processed_train_metadata.json"
IMAGE_NPY_DIR = "processed_data/train"
SAVE_DIR = "embeddings/train"
BATCH_SIZE = 32  # RAM-safe batch size
MAX_IMAGES = 3  # Max images per study (1-3 images in dataset)

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Load processed metadata
with open(TEXT_JSON_PATH, "r") as f:
    metadata = json.load(f)

def extract_study_uid(full_id: str) -> str:
    return full_id.split("_s")[-1]

# Convert metadata to list of (StudyUID, text)
items = [
    (extract_study_uid(full_id), text)
    for full_id, text in metadata.items()
    if os.path.isfile(os.path.join(IMAGE_NPY_DIR, f"{extract_study_uid(full_id)}.npy"))
]

# === Processing in Batches ===
for i in tqdm(range(0, len(items), BATCH_SIZE), desc="Generating Embeddings"):
    batch = items[i:i+BATCH_SIZE]
    study_uids = [uid for uid, _ in batch]
    texts = [text for _, text in batch]

    # Load and process image arrays
    image_arrays = []
    image_masks = []
    valid_uids = []
    valid_texts = []
    for uid, text in zip(study_uids, texts):
        npy_path = os.path.join(IMAGE_NPY_DIR, f"{uid}.npy")
        try:
            arr = np.load(npy_path)  # shape: [N, 224, 224, 3]
            if arr.ndim != 4 or arr.shape[1:3] != (224, 224):
                print(f"Skipping {uid}: Invalid shape {arr.shape}")
                continue
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            continue

        # Limit to MAX_IMAGES (use last N for temporal order, assuming chronological)
        arr = arr[-MAX_IMAGES:]
        N = arr.shape[0]

        # Pad with zeros if fewer than MAX_IMAGES
        if N < MAX_IMAGES:
            pad_shape = (MAX_IMAGES - N, 224, 224, 3)
            arr = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)

        # Transpose to [MAX_IMAGES, 3, 224, 224]
        arr = np.transpose(arr, (0, 3, 1, 2))
        image_arrays.append(arr)

        # Create mask for valid images
        mask = np.ones(MAX_IMAGES, dtype=np.long)
        mask[N:] = 0
        image_masks.append(mask)

        # Track valid studies
        valid_uids.append(uid)
        valid_texts.append(text)

    if not image_arrays:  # Skip empty batches
        print(f"Skipping batch {i // BATCH_SIZE}: No valid images")
        continue

    # Stack into [B, MAX_IMAGES, 3, 224, 224] tensor
    image_tensors = torch.tensor(np.stack(image_arrays)).float().to(device)

    # Normalize images (ImageNet mean/std for ResNet-50 in BioViL-T)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensors = normalize(image_tensors)

    # Create image mask tensor
    image_mask = torch.tensor(np.stack(image_masks)).long().to(device)

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
            # Note: image_mask not passed to model (BioViL-T doesn't explicitly support it)
        )

        # Debug output structure for first batch
        if i == 0:
            print("Model output keys:", list(outputs.keys()))

        # Extract embeddings (adjust based on actual output keys if needed)
        text_embeds = outputs.text_embeds  # [B, D]
        image_embeds = outputs.image_embeds  # [B, D]

    # Save embeddings
    save_path = os.path.join(SAVE_DIR, f"batch_{i // BATCH_SIZE:04d}.pt")
    torch.save({
        "study_uids": valid_uids,
        "text_embeds": text_embeds.cpu(),
        "image_embeds": image_embeds.cpu(),
        "image_mask": image_mask.cpu()  # Save for downstream use
    }, save_path)

    # Cleanup
    del text_embeds, image_embeds, input_ids, attention_mask, image_tensors, image_mask
    torch.cuda.empty_cache()

print(f"[INFO] All embeddings saved in: {SAVE_DIR}")