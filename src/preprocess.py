import nibabel as nib
import numpy as np
import os

nifti_folder = os.path.join("..", "data", "nifti")
processed_folder = os.path.join("..", "data", "nifti_windowed")
os.makedirs(processed_folder, exist_ok=True)

# Cardiac Window: Level 40, Width 400 -> Range [-160, 240]
MIN_HU = -160
MAX_HU = 240

for file_name in os.listdir(nifti_folder):
    if not file_name.endswith(".nii.gz"): continue
    
    # Load
    img = nib.load(os.path.join(nifti_folder, file_name))
    data = img.get_fdata()
    
    # Apply Windowing
    data = np.clip(data, MIN_HU, MAX_HU)
    
    # Normalize to [0, 1] for the Neural Network
    data = (data - MIN_HU) / (MAX_HU - MIN_HU)
    
    # Save
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, os.path.join(processed_folder, file_name))
    print(f"Windowed & Normalized: {file_name}")

print("\n Task 1 Preprocessing (HU Windowing) complete!")