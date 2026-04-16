"""Shared MONAI transforms for Merlin LP feature extraction."""
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

from merlin.data.monai_transforms import MaskCenterCropd


ROI_SIZE = (224, 224, 160)
PIXDIM = (1.5, 1.5, 3)
INTENSITY_RANGE = dict(a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)


def build_roi_transform():
    """Crop a 224×224×160 region centered on the binary fg_mask (label=1)."""
    return Compose([
        LoadImaged(keys=["image", "mask", "mask_original"]),
        EnsureChannelFirstd(keys=["image", "mask", "mask_original"]),
        Orientationd(keys=["image", "mask", "mask_original"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=PIXDIM, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], **INTENSITY_RANGE),
        MaskCenterCropd(keys=["image"], mask_key="mask", roi_size=list(ROI_SIZE), fg_labels=[1]),
        DeleteItemsd(keys=["mask", "mask_original"]),
        ToTensord(keys=["image"]),
    ])


def build_non_roi_transform():
    """Center-crop a 224×224×160 region (no mask available)."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=PIXDIM, mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], **INTENSITY_RANGE),
        SpatialPadd(keys=["image"], spatial_size=list(ROI_SIZE)),
        CenterSpatialCropd(keys=["image"], roi_size=list(ROI_SIZE)),
        ToTensord(keys=["image"]),
    ])
