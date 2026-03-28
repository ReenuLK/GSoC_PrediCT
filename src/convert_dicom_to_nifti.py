import os
import dicom2nifti
import logging

# 1. Setup paths
# raw_dicom_path is where the DICOM files are stored. Each patient should have their own folder.
raw_dicom_path = r"C:\COCA_Dataset\cocacoronarycalciumandchestcts-2\Gated_release_final\patient" 
nifti_out_path = os.path.join("..", "data", "nifti")

os.makedirs(nifti_out_path, exist_ok=True)

# 2. Get patient folders
patients = [p for p in os.listdir(raw_dicom_path) if os.path.isdir(os.path.join(raw_dicom_path, p))]

print(f"Found {len(patients)} patients. Starting conversion...")

# 3. Conversion Loop
for p_id in patients[:51]: 
    input_folder = os.path.join(raw_dicom_path, p_id)
    output_file = os.path.join(nifti_out_path, f"case_{p_id}.nii.gz")
    
    if os.path.exists(output_file):
        print(f" Skipping {p_id}, already exists.")
        continue

    try:
        # reorient_nifti=True ensures the heart is 'upright' for the AI
        dicom2nifti.dicom_series_to_nifti(input_folder, output_file, reorient_nifti=True)
        print(f" Finished Patient {p_id}")
    except Exception as e:
        print(f" Error on Patient {p_id}: {e}")

print("\n Phase 1 Test Complete. Check GSoC_PrediCT/data/nifti")