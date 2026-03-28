import os
import nibabel as nib
import numpy as np
import pandas as pd

# --- VERIFIED PATHS ---
IMAGE_DIR = r"C:\GSoc_PrediCT\data\nifti_windowed"
LABEL_ROOT = r"C:\GSoc_PrediCT\data\labels"

stats = []

print("Reading 50 patients from subdirectories... ")

# Walk through the labels directory to find nested .nii.gz files
for root, dirs, files in os.walk(LABEL_ROOT):
    for file in files:
        if file == "combined_heart.nii.gz":  # Match your merged filename
            lbl_path = os.path.join(root, file)
            
            # Extract case name from the folder name (e.g., 'case_1')
            case_folder = os.path.basename(root) 
            img_filename = f"{case_folder}.nii.gz"
            img_path = os.path.join(IMAGE_DIR, img_filename)
            
            if os.path.exists(img_path):
                try:
                    # Load Label
                    lbl = nib.load(lbl_path)
                    data_lbl = lbl.get_fdata()
                    header = lbl.header
                    
                    # Load Image
                    img = nib.load(img_path)
                    data_img = img.get_fdata()
                    
                    # 1. Volume Calculation
                    vox_size = np.prod(header.get_zooms())
                    heart_vol_ml = (np.count_nonzero(data_lbl) * vox_size) / 1000
                    
                    # 2. Intensity Stats
                    heart_intensities = data_img[data_lbl > 0]
                    mean_hu = np.mean(heart_intensities) if heart_intensities.size > 0 else 0
                    
                    stats.append({
                        "Case": case_folder,
                        "Volume_mL": heart_vol_ml,
                        "Mean_HU": mean_hu,
                        "Spacing": header.get_zooms()
                    })
                    print(f" Processed {case_folder}")
                    
                except Exception as e:
                    print(f" Error processing {case_folder}: {e}")
            else:
                print(f" Image not found for {case_folder} (Checked: {img_path})")

if stats:
    df = pd.DataFrame(stats)
    print("\n" + "="*40)
    print(" FINAL DATASET STATISTICS (N=50)")
    print("="*40)
    print(f"Mean Heart Volume:    {df['Volume_mL'].mean():.2f} mL")
    print(f"Volume Std Dev:       {df['Volume_mL'].std():.2f} mL")
    print(f"Mean Intensity:       {df['Mean_HU'].mean():.2f} HU")
    print(f"Common Spacing:       {stats[0]['Spacing']} mm")
    print("="*40)
else:
    print("No data collected. Please verify if 'combined_heart.nii.gz' exists inside the case folders.")