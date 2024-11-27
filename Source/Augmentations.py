
import albumentations as A


def get_augmentations(image_width: int, image_height: int):
    base_transforms = A.Compose([
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=4, p=0.6),
            A.VerticalFlip(p=0.5)
        ])

    color_transforms = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5)
        ])

    val_transforms = A.Compose([
            A.Resize(height=image_height, width=image_width),
        ])

    return base_transforms, color_transforms, val_transforms
