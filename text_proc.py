# import json
# import re

# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = re.sub(r'\s+', ' ', text)  # remove excessive whitespace
#     return text.strip()

# def ensure_ends_with_period(text):
#     if not text:
#         return text
#     text = text.strip()
#     if text[-1] not in {".", "!", "?"}:
#         text += "."
#     return text

# def strip_field_prefix(text, field):
#     """
#     Removes leading field name like 'Comparison:', 'Indication:', etc.
#     """
#     text = text.strip()
#     field_prefix = field.lower() + ":"
#     if text.lower().startswith(field_prefix):
#         return text[len(field_prefix):].strip()
#     return text

# def format_patient_age(age_str):
#     if not age_str:
#         return "Patient Age: Unknown."
#     age_str = age_str.strip().upper()
#     if age_str.endswith("Y"):
#         return f"Patient Age: {int(age_str[:-1])} years."
#     elif age_str.endswith("M"):
#         return f"Patient Age: {int(age_str[:-1])} months."
#     elif age_str.endswith("D"):
#         return f"Patient Age: {int(age_str[:-1])} days."
#     else:
#         return "Patient Age: Unknown."

# def format_patient_sex(sex_str):
#     if not sex_str:
#         return "Patient Sex: Unknown."
#     sex_str = sex_str.upper()
#     mapping = {"M": "Male", "F": "Female", "O": "Unknown"}
#     return f"Patient Sex: {mapping.get(sex_str, 'Unknown')}."

# def construct_raw_text(study):
#     parts = []

#     # Patient Age and Sex
#     age_str = study.get("PatientAge", "")
#     sex_str = study.get("PatientSex", "")
#     parts.append(format_patient_age(age_str))
#     parts.append(format_patient_sex(sex_str))

#     # Existing fields
#     for field in ["Indication", "Findings", "Impression", "Comparison"]:
#         value = study.get(field, "")
#         if value and isinstance(value, str) and value.lower() not in {"none", "none."}:
#             cleaned = clean_text(value)
#             cleaned = strip_field_prefix(cleaned, field)
#             cleaned = ensure_ends_with_period(cleaned)
#             parts.append(f"{field}: {cleaned}")

#     return " ".join(parts)

# def process_all_studies(metadata_dict):
#     processed_texts = {}
#     for study_id, study_data in metadata_dict.items():
#         processed_texts[study_id] = construct_raw_text(study_data)
#     print(f"[INFO] Processed {len(processed_texts)} studies.")
#     return processed_texts

# # def save_texts_to_json(processed_texts, save_path="metadata/processed/processed_train_metadata.json"):
# #     with open(save_path, 'w') as f:
# #         json.dump(processed_texts, f, indent=2)
# #     print(f"[INFO] Saved processed texts to '{save_path}'.")

# # with open("metadata/updated/updated_train_metadata.json") as f:
# #     metadata_dict = json.load(f)


# # def save_texts_to_json(processed_texts, save_path="metadata/processed/processed_test_metadata.json"):
# #     with open(save_path, 'w') as f:
# #         json.dump(processed_texts, f, indent=2)
# #     print(f"[INFO] Saved processed texts to '{save_path}'.")

# # with open("metadata/updated/updated_test_metadata.json") as f:
# #     metadata_dict = json.load(f)


# def save_texts_to_json(processed_texts, save_path="metadata/processed/processed_valid_metadata.json"):
#     with open(save_path, 'w') as f:
#         json.dump(processed_texts, f, indent=2)
#     print(f"[INFO] Saved processed texts to '{save_path}'.")

# with open("metadata/updated/updated_valid_metadata.json") as f:
#     metadata_dict = json.load(f)

# processed_texts = process_all_studies(metadata_dict)
# save_texts_to_json(processed_texts)




import json
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)  # remove excessive whitespace
    return text.strip()

def ensure_ends_with_period(text):
    if not text:
        return text
    text = text.strip()
    if text[-1] not in {".", "!", "?"}:
        text += "."
    return text

def strip_field_prefix(text, field):
    """
    Removes leading field name like 'Comparison:', 'Indication:', etc.
    """
    text = text.strip()
    field_prefix = field.lower() + ":"
    if text.lower().startswith(field_prefix):
        return text[len(field_prefix):].strip()
    return text

def format_patient_age(age_str):
    if not age_str:
        return "Patient Age: Unknown."
    age_str = age_str.strip().upper()
    if age_str.endswith("Y"):
        return f"Patient Age: {int(age_str[:-1])} years."
    elif age_str.endswith("M"):
        return f"Patient Age: {int(age_str[:-1])} months."
    elif age_str.endswith("D"):
        return f"Patient Age: {int(age_str[:-1])} days."
    else:
        return "Patient Age: Unknown."

def format_patient_sex(sex_str):
    if not sex_str:
        return "Patient Sex: Unknown."
    sex_str = sex_str.upper()
    mapping = {"M": "Male", "F": "Female", "O": "Unknown"}
    return f"Patient Sex: {mapping.get(sex_str, 'Unknown')}."

def construct_raw_text(study):
    parts = []

    # Patient Age and Sex
    age_str = study.get("PatientAge", "")
    sex_str = study.get("PatientSex", "")
    if age_str:  # Only add age if it’s not empty or null
        parts.append(format_patient_age(age_str))
    parts.append(format_patient_sex(sex_str))

    # Existing fields
    for field in ["Indication", "Findings", "Impression", "Comparison"]:
        value = study.get(field, "")
        if value and isinstance(value, str) and value.lower() not in {"none", "none."}:
            cleaned = clean_text(value)
            cleaned = strip_field_prefix(cleaned, field)
            cleaned = ensure_ends_with_period(cleaned)
            parts.append(f"{field}: {cleaned}")

    return " ".join(parts)

def process_all_studies(metadata_dict):
    processed_texts = {}
    for study_id, study_data in metadata_dict.items():
        processed_texts[study_id] = construct_raw_text(study_data)
    print(f"[INFO] Processed {len(processed_texts)} studies.")
    return processed_texts

def save_texts_to_json(processed_texts, save_path="metadata/processed/processed_train_metadata.json"):
    with open(save_path, 'w') as f:
        json.dump(processed_texts, f, indent=2)
    print(f"[INFO] Saved processed texts to '{save_path}'.")

with open("metadata/updated/updated_train_metadata.json") as f:
    metadata_dict = json.load(f)


# def save_texts_to_json(processed_texts, save_path="metadata/processed/processed_test_metadata.json"):
#     with open(save_path, 'w') as f:
#         json.dump(processed_texts, f, indent=2)
#     print(f"[INFO] Saved processed texts to '{save_path}'.")

# with open("metadata/updated/updated_test_metadata.json") as f:
#     metadata_dict = json.load(f)

# def save_texts_to_json(processed_texts, save_path="metadata/processed/processed_valid_metadata.json"):
#     with open(save_path, 'w') as f:
#         json.dump(processed_texts, f, indent=2)
#     print(f"[INFO] Saved processed texts to '{save_path}'.")

# with open("metadata/updated/updated_valid_metadata.json") as f:
#     metadata_dict = json.load(f)

processed_texts = process_all_studies(metadata_dict)
save_texts_to_json(processed_texts)