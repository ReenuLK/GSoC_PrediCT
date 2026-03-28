from monai.transforms import (
    Compose, LoadImaged, Spacingd, ScaleIntensityRanged, 
    RandAffined, RandGaussianNoised, EnsureTyped, Resized
)
from monai.data import Dataset, DataLoader, pad_list_data_collate

def get_train_loader(data_dicts):
    train_transforms = Compose([
        # 1. Load the files (removed spatial_size from here)
        LoadImaged(keys=["image", "label"]),
        
        # 2. Resizing: THIS FIXES THE BATCH SIZE ERROR
        # Every 3D volume will now be exactly 128x128x128
        Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
        
        # 3. Windowing: Focus only on Heart tissue
        ScaleIntensityRanged(
            keys=["image"], a_min=-160, a_max=240, 
            b_min=0.0, b_max=1.0, clip=True
        ),
        
        # 4. Resampling: 1mm isotropic
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.0, 1.0, 1.0), 
            mode=("bilinear", "nearest")
        ),
        
        # 5. Augmentation
        RandAffined(
            keys=["image", "label"], prob=0.5,
            rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)
        ),
        RandGaussianNoised(keys=["image"], prob=0.2),
        
        # 6. Final Prep
        EnsureTyped(keys=["image", "label"])
    ])
    
    ds = Dataset(data=data_dicts, transform=train_transforms)
    
    # collate_fn=pad_list_data_collate is the "safety net" for batching
    return DataLoader(
        ds, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=pad_list_data_collate
    )
