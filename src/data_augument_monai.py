from monai.transforms import (
    Compose, LoadImaged, Spacingd, Orientationd, 
    ScaleIntensityRanged, RandAffined, RandGaussianNoised, 
    EnsureTyped, Resized
)
from monai.data import Dataset, DataLoader, pad_list_data_collate

def get_train_loader(data_dicts):
    """
    Optimized Data Loader for GSoC Project 1: Heart-Box Model.
    Sequence: Load -> Orient -> Intensity -> Physics -> Resizing -> Augment
    """
    train_transforms = Compose([
        # 1. Load the 3D Volumes
        LoadImaged(keys=["image", "label"]),
        
        # 2. Standardize Orientation (RAS: Right, Anterior, Superior)
        # Crucial for the model to know which way is 'Up' across different scans
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # 3. Windowing: Focus on Heart tissue (Soft Tissue Window)
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=-160, a_max=240, 
            b_min=0.0, b_max=1.0, clip=True
        ),
        
        # 4. Resampling: Standardize to 1mm isotropic voxels
        # This fixes physical proportions BEFORE we change the matrix size
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.0, 1.0, 1.0), 
            mode=("bilinear", "nearest")
        ),
        
        # 5. Geometry Fix: Force a 128x128x128 cube for the GPU
        # This prevents the 'tensor size mismatch' error in the DataLoader
        Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
        
        # 6. Data Augmentation (The 'Robustness' layer)
        RandAffined(
            keys=["image", "label"], prob=0.5,
            rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)
        ),
        RandGaussianNoised(keys=["image"], prob=0.2),
        
        # 7. Final Prep for CUDA
        EnsureTyped(keys=["image", "label"])
    ])
    
    # Create the dataset
    ds = Dataset(data=data_dicts, transform=train_transforms)
    
    # Return the loader with the safety collate function
    return DataLoader(
        ds, 
        batch_size=2, 
        shuffle=True, 
        num_workers=2,
        collate_fn=pad_list_data_collate
    )
