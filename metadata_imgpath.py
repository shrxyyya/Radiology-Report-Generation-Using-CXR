import json
import os

# Load existing metadata
with open('metadata/updated/updated_train_metadata.json', 'r') as f:
    metadata = json.load(f)

# Define base path to processed images
processed_base_path = "processed_data/train"

# Track missing files
missing = []
updated = 0

# Update metadata
for study_key, study_data in metadata.items():
    study_uid = study_data.get("StudyInstanceUid")
    if study_uid:
        npy_path = os.path.join(processed_base_path, f"{study_uid}.npy")
        
        if os.path.exists(npy_path):
            study_data["ProcImagePath"] = npy_path
            updated += 1
        else:
            missing.append(study_uid)
            # Optional: Remove the field if it shouldn't exist for missing files
            # study_data.pop("ProcImagePath", None)

# Save updated metadata
with open('metadata/updated_train_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Report
print(f"\n✅ Updated metadata for {updated} studies.")
if missing:
    print(f"⚠️ {len(missing)} studies in the metadata have no corresponding processed .npy file:")
    for uid in missing:
        print(f" - {uid}")




# import json
# import os

# # Load existing metadata
# with open('metadata/updated/updated_test_metadata.json', 'r') as f:
#     metadata = json.load(f)

# # Define base path to processed images
# processed_base_path = "processed_data/test"

# # Track missing files
# missing = []
# updated = 0

# # Update metadata
# for study_key, study_data in metadata.items():
#     study_uid = study_data.get("StudyInstanceUid")
#     if study_uid:
#         npy_path = os.path.join(processed_base_path, f"{study_uid}.npy")
        
#         if os.path.exists(npy_path):
#             study_data["ProcImagePath"] = npy_path
#             updated += 1
#         else:
#             missing.append(study_uid)
#             # Optional: Remove the field if it shouldn't exist for missing files
#             # study_data.pop("ProcImagePath", None)

# # Save updated metadata
# with open('metadata/updated_test_metadata.json', 'w') as f:
#     json.dump(metadata, f, indent=2)

# # Report
# print(f"\n✅ Updated metadata for {updated} studies.")
# if missing:
#     print(f"⚠️ {len(missing)} studies in the metadata have no corresponding processed .npy file:")
#     for uid in missing:
#         print(f" - {uid}")



# import json
# import os

# # Load existing metadata
# with open('metadata/updated/updated_valid_metadata.json', 'r') as f:
#     metadata = json.load(f)

# # Define base path to processed images
# processed_base_path = "processed_data/valid"

# # Track missing files
# missing = []
# updated = 0

# # Update metadata
# for study_key, study_data in metadata.items():
#     study_uid = study_data.get("StudyInstanceUid")
#     if study_uid:
#         npy_path = os.path.join(processed_base_path, f"{study_uid}.npy")
        
#         if os.path.exists(npy_path):
#             study_data["ProcImagePath"] = npy_path
#             updated += 1
#         else:
#             missing.append(study_uid)
#             # Optional: Remove the field if it shouldn't exist for missing files
#             # study_data.pop("ProcImagePath", None)

# # Save updated metadata
# with open('metadata/updated_valid_metadata.json', 'w') as f:
#     json.dump(metadata, f, indent=2)

# # Report
# print(f"\n✅ Updated metadata for {updated} studies.")
# if missing:
#     print(f"⚠️ {len(missing)} studies in the metadata have no corresponding processed .npy file:")
#     for uid in missing:
#         print(f" - {uid}")