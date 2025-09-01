import os
import shutil
import json
from tqdm import tqdm

# ✅ Source of the existing .npy files
source_data_dir = 'processed_data'

# ✅ Where to store the train split
processed_data_dir = 'processed_data_new'
train_output_dir = os.path.join(processed_data_dir, 'train')
os.makedirs(train_output_dir, exist_ok=True)

# ✅ Load train metadata
metadata_path = 'metadata/updated/updated_train_metadata.json'
with open(metadata_path, 'r') as f:
    train_metadata = json.load(f)

# ✅ Setup
missing = []
copied = 0
show_paths = False  # Set to True if you want to print each path

# ✅ Copy with progress bar
for study_key, study_data in tqdm(train_metadata.items(), desc="Copying .npy files", unit="file"):
    study_uid = study_data.get("StudyInstanceUid")
    if not study_uid:
        continue

    # Build source and destination paths
    npy_source_path = os.path.join(source_data_dir, study_uid, f"{study_uid}.npy")
    npy_dest_path = os.path.join(train_output_dir, f"{study_uid}.npy")

    if show_paths:
        print(f"Looking for file: {npy_source_path}")

    if os.path.exists(npy_source_path):
        try:
            shutil.copy2(npy_source_path, npy_dest_path)
            copied += 1
        except Exception as e:
            print(f"Error copying {study_uid}: {e}")
            missing.append(study_uid)
    else:
        missing.append(study_uid)

# ✅ Final report
print(f"\n✅ Copied {copied} files to '{train_output_dir}'.")
if missing:
    print(f"⚠️ {len(missing)} files were missing or failed to copy:")
    for uid in missing:
        print(f" - {uid}")



# import os
# import shutil
# import json
# from tqdm import tqdm

# # ✅ Source of the existing .npy files
# source_data_dir = 'processed_data'

# # ✅ Where to store the train split
# processed_data_dir = 'processed_data_new'
# test_output_dir = os.path.join(processed_data_dir, 'test')
# os.makedirs(test_output_dir, exist_ok=True)

# # ✅ Load train metadata
# metadata_path = 'metadata/updated/updated_test_metadata.json'
# with open(metadata_path, 'r') as f:
#     test_metadata = json.load(f)

# # ✅ Setup
# missing = []
# copied = 0

# # ✅ Copy with progress bar
# for study_key, study_data in tqdm(test_metadata.items(), desc="Copying .npy files", unit="file"):
#     study_uid = study_data.get("StudyInstanceUid")
#     if not study_uid:
#         continue

#     # Build source and destination paths
#     npy_source_path = os.path.join(source_data_dir, study_uid, f"{study_uid}.npy")
#     npy_dest_path = os.path.join(test_output_dir, f"{study_uid}.npy")

#     if os.path.exists(npy_source_path):
#         try:
#             shutil.copy2(npy_source_path, npy_dest_path)
#             copied += 1
#         except Exception as e:
#             print(f"Error copying {study_uid}: {e}")
#             missing.append(study_uid)
#     else:
#         missing.append(study_uid)

# # ✅ Final report
# print(f"\n✅ Copied {copied} files to '{test_output_dir}'.")
# if missing:
#     print(f"⚠️ {len(missing)} files were missing or failed to copy:")
#     for uid in missing:
#         print(f" - {uid}")




# import os
# import shutil
# import json
# from tqdm import tqdm

# # ✅ Source of the existing .npy files
# source_data_dir = 'processed_data'

# # ✅ Where to store the train split
# processed_data_dir = 'processed_data_new'
# valid_output_dir = os.path.join(processed_data_dir, 'valid')
# os.makedirs(valid_output_dir, exist_ok=True)

# # ✅ Load train metadata
# metadata_path = 'metadata/updated/updated_valid_metadata.json'
# with open(metadata_path, 'r') as f:
#     valid_metadata = json.load(f)

# # ✅ Setup
# missing = []
# copied = 0

# # ✅ Copy with progress bar
# for study_key, study_data in tqdm(valid_metadata.items(), desc="Copying .npy files", unit="file"):
#     study_uid = study_data.get("StudyInstanceUid")
#     if not study_uid:
#         continue

#     # Build source and destination paths
#     npy_source_path = os.path.join(source_data_dir, study_uid, f"{study_uid}.npy")
#     npy_dest_path = os.path.join(valid_output_dir, f"{study_uid}.npy")

#     if os.path.exists(npy_source_path):
#         try:
#             shutil.copy2(npy_source_path, npy_dest_path)
#             copied += 1
#         except Exception as e:
#             print(f"Error copying {study_uid}: {e}")
#             missing.append(study_uid)
#     else:
#         missing.append(study_uid)

# # ✅ Final report
# print(f"\n✅ Copied {copied} files to '{valid_output_dir}'.")
# if missing:
#     print(f"⚠️ {len(missing)} files were missing or failed to copy:")
#     for uid in missing:
#         print(f" - {uid}")
