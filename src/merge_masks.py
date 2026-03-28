import os
import nibabel as nib
import numpy as np

labels_dir = r"C:\GSoc_PrediCT\data\labels"

# Mapping of filename to Integer ID
# These IDs are standard for many heart models
CHAMBER_MAP = {
    "heart_ventricle_left.nii.gz": 1,
    "heart_ventricle_right.nii.gz": 2,
    "heart_atrium_left.nii.gz": 3,
    "heart_atrium_right.nii.gz": 4,
    "heart_myocardium.nii.gz": 5
}

for patient_id in os.listdir(labels_dir):
    p_path = os.path.join(labels_dir, patient_id)
    if not os.path.isdir(p_path): continue
    
    output_file = os.path.join(p_path, "combined_heart.nii.gz")
    
    # Skip if already merged
    if os.path.exists(output_file): continue
    
    print(f"Merging chambers for {patient_id}...")
    
    master_data = None
    affine = None
    
    for filename, label_id in CHAMBER_MAP.items():
        file_path = os.path.join(p_path, filename)
        
        if os.path.exists(file_path):
            img = nib.load(file_path)
            data = img.get_fdata()
            
            if master_data is None:
                master_data = np.zeros(data.shape, dtype=np.uint8)
                affine = img.affine
            
            # Place the label ID where the mask is 1
            master_data[data > 0.5] = label_id

    if master_data is not None:
        combined_img = nib.Nifti1Image(master_data, affine)
        nib.save(combined_img, output_file)
        print(f" Created {output_file}")

print("\n All current labels merged!")