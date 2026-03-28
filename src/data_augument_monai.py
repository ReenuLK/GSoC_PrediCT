from monai.transforms import (
    Compose, LoadImaged, Spacingd, ScaleIntensityRanged, 
    RandAffined, RandGaussianNoised, EnsureTyped
)
from monai.data import Dataset, DataLoader

def get_train_loader(data_dicts):
    # This list defines the "Recipe" for the data
    train_transforms = Compose([
        # 1. Load the files
        LoadImaged(keys=["image", "label"]),
        
        # 2. Windowing: Focus only on Heart tissue
        ScaleIntensityRanged(
            keys=["image"], a_min=-160, a_max=240, 
            b_min=0.0, b_max=1.0, clip=True
        ),
        
        # 3. Resampling: Make every voxel a 1mm x 1mm x 1mm cube
        # This prevents the heart from looking "squashed"
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.0, 1.0, 1.0), 
            mode=("bilinear", "nearest")
        ),
        
        # 4. Augmentation: Randomly tilt and add noise (The "Data Multiplier")
        RandAffined(
            keys=["image", "label"], prob=0.5,
            rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)
        ),
        RandGaussianNoised(keys=["image"], prob=0.2),
        
        # 5. Convert to format the GPU understands
        EnsureTyped(keys=["image", "label"])
    ])
    
    # Create the actual loader
    ds = Dataset(data=data_dicts, transform=train_transforms)
    return DataLoader(ds, batch_size=2, shuffle=True)