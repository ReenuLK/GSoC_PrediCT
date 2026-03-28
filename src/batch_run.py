import os
import subprocess
import sys

# Settings
NIFTI_DIR = r"C:\GSoc_PrediCT\data\nifti"
OUTPUT_DIR = r"C:\GSoc_PrediCT\data\labels"
BATCH_LIMIT = 10  # Process 10 at a time

# 1. Identify which patients still need processing
all_niftis = [f for f in os.listdir(NIFTI_DIR) if f.endswith(".nii.gz")]
to_process = []

for f in all_niftis:
    p_id = f.replace(".nii.gz", "")
    p_folder = os.path.join(OUTPUT_DIR, p_id)
    
    # CHECK: Does the folder exist? Does it have the 5 chamber files?
    # We check for > 4 files to ensure the 5 masks (or the merged file) are there.
    if not os.path.exists(p_folder) or len(os.listdir(p_folder)) < 5:
        to_process.append(f)

# 2. Limit the batch to your BATCH_LIMIT
current_batch = to_process[:BATCH_LIMIT]

print(f" Total patients missing labels: {len(to_process)}")
print(f" Starting current batch of {len(current_batch)}...")

# 3. Execution Loop
for file in current_batch:
    p_id = file.replace(".nii.gz", "")
    in_p = os.path.join(NIFTI_DIR, file)
    out_p = os.path.join(OUTPUT_DIR, p_id)
    
    print(f"\n[RUNNING] ---> {p_id}")
    
    cmd = [
        sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator",
        "-i", in_p,
        "-o", out_p,
        "-ta", "heartchambers_highres"
    ]
    
    try:
        # We use check=True so we know if it actually finished
        subprocess.run(cmd, check=True)
        print(f" COMPLETED: {p_id}")
    except Exception as e:
        print(f" ERROR on {p_id}: {e}")

print(f"\n Batch of {len(current_batch)} finished! Check your labels folder.")