# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from pathlib import Path
# from tqdm import tqdm
# import gc

# # --- Configuration ---
# output_dir = 'normalised_processed_data'

# # --- Collect all .npy files ---
# npy_files = []
# for root, _, files in os.walk(output_dir):
#     for file in files:
#         if file.endswith('.npy'):
#             npy_files.append(os.path.join(root, file))

# print(f"📁 Found {len(npy_files)} .npy files in {output_dir}")

# # --- Inspection results ---
# results = []
# issues = {'negatives': [], 'nans': [], 'excessive_zeros': []}

# # --- Inspect each .npy file ---
# for file_path in tqdm(npy_files, desc="Inspecting .npy files"):
#     # Load the .npy file with memory mapping
#     try:
#         tensor = np.load(file_path, mmap_mode='r')
#     except Exception as e:
#         print(f"⚠️ Error loading file {file_path}: {e}")
#         continue

#     # Extract study ID from file path
#     study_id = Path(file_path).parent.name

#     # Collect basic information
#     info = {
#         'file_path': file_path,
#         'study_id': study_id,
#         'shape': tensor.shape,
#         'dtype': tensor.dtype,
#         'min_value': np.min(tensor),
#         'max_value': np.max(tensor),
#         'mean_value': np.mean(tensor),
#         'std_value': np.std(tensor),
#         'nan_count': np.isnan(tensor).sum(),
#         'negative_count': np.sum(tensor < 0),
#         'zero_count': np.sum(tensor == 0)
#     }

#     # Check for issues
#     if info['negative_count'] > 0:
#         issues['negatives'].append((file_path, info['negative_count']))
#     if info['nan_count'] > 0:
#         issues['nans'].append((file_path, info['nan_count']))
#     if info['zero_count'] > tensor.size * 0.5:
#         issues['excessive_zeros'].append((file_path, info['zero_count'] / tensor.size))

#     results.append(info)

#     # Print detailed information
#     print(f"\n📄 File: {file_path}")
#     print(f"Study ID: {study_id}")
#     print(f"Tensor shape: {tensor.shape}")
#     print(f"Data type: {tensor.dtype}")
#     print(f"Min value: {info['min_value']}")
#     print(f"Max value: {info['max_value']}")
#     print(f"Mean value: {info['mean_value']}")
#     print(f"Standard deviation: {info['std_value']}")
#     print(f"NaN values: {info['nan_count']}")
#     print(f"Negative values: {info['negative_count']}")
#     print(f"Zero values: {info['zero_count']} ({info['zero_count'] / tensor.size:.2%} of tensor)")

#     # Print sample values (first 5x5 pixels of first image, first channel)
#     if tensor.ndim == 4:
#         print("\nSample values (first 5x5 pixels of first image, first channel):")
#         print(tensor[0, :5, :5, 0])

#     # Visualize first image (optional, comment out if not needed)
#     # if tensor.ndim == 4:
#     #     plt.figure(figsize=(6, 6))
#     #     plt.imshow(np.array(tensor[0, :, :, 0]), cmap='gray')  # Convert to array if needed for plotting
#     #     plt.title(f'First Image of Study {study_id}')
#     #     plt.colorbar()
#     #     plt.show()

#     # Plot pixel value distribution (optional, comment out if not needed)
#     # if tensor.ndim == 4:
#     #     plt.figure(figsize=(6, 4))
#     #     plt.hist(np.ravel(tensor), bins=50, range=(0, 1))
#     #     plt.title(f'Pixel Value Distribution for Study {study_id}')
#     #     plt.xlabel('Pixel Value')
#     #     plt.ylabel('Frequency')
#     #     plt.show()

#     # Clean up memory
#     del tensor
#     gc.collect()

# # --- Summary report ---
# print("\n✅ Inspection complete.")
# print(f"📁 Total .npy files inspected: {len(results)}")
# print(f"🛑 Issues found:")
# print(f"  - Files with negative values: {len(issues['negatives'])}")
# for file_path, count in issues['negatives']:
#     print(f"    - {file_path}: {count} negative values")
# print(f"  - Files with NaN values: {len(issues['nans'])}")
# for file_path, count in issues['nans']:
#     print(f"    - {file_path}: {count} NaN values")
# print(f"  - Files with >50% zero values: {len(issues['excessive_zeros'])}")
# for file_path, ratio in issues['excessive_zeros']:
#     print(f"    - {file_path}: {ratio:.2%} zero values")



import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- Configuration ---
output_dir = 'normalised_processed_data'
# Specify the .npy file to inspect (replace with the desired study ID or file path)
file_path = 'normalised_processed_data/1.2.826.0.1.3680043.8.498.10000633072033160475771834275511027574/1.2.826.0.1.3680043.8.498.10000633072033160475771834275511027574.npy'
plot_output_dir = 'plot_outputs'  # Directory to save plots

# --- Create plot output directory ---
Path(plot_output_dir).mkdir(exist_ok=True)

# --- Load the .npy file ---
try:
    tensor = np.load(file_path, mmap_mode='r')
except Exception as e:
    print(f"⚠️ Error loading file {file_path}: {e}")
    exit()

# --- Extract study ID and validate tensor ---
study_id = Path(file_path).parent.name
if tensor.ndim != 4:
    print(f"⚠️ Error: Expected 4D tensor (num_images, 224, 224, 3), got shape {tensor.shape}")
    exit()

# --- Plot pixel value distribution for the first image ---
plt.figure(figsize=(6, 4))
plt.hist(tensor[0, :, :, 0].flatten(), bins=50, range=(0, 1), density=True)
plt.title(f'Pixel Value Distribution for First Image of Study {study_id}')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency (Normalized)')
plt.grid(True, alpha=0.3)
# Save the histogram
histogram_path = os.path.join(plot_output_dir, f'{study_id}_distribution.png')
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 Saved pixel value distribution to {histogram_path}")

# --- Visualize the first image ---
plt.figure(figsize=(6, 6))
plt.imshow(tensor[0, :, :, 0], cmap='gray')
plt.title(f'First Image of Study {study_id}')
plt.colorbar()
plt.axis('off')
# Save the image visualization
image_path = os.path.join(plot_output_dir, f'{study_id}_image.png')
plt.savefig(image_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"🖼️ Saved image visualization to {image_path}")