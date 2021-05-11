import albumentations
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(img_size=512):
    return albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        albumentations.HueSaturationValue(p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
        albumentations.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
        albumentations.CoarseDropout(p=0.5),
        albumentations.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        ToTensorV2(p=1.0)
    ])

def get_valid_transforms(img_size=512):

    return albumentations.Compose([
        albumentations.Resize(img_size, img_size, always_apply=True),
        albumentations.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        ToTensorV2(p=1.0)
    ])
