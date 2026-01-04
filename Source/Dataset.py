import albumentations as A
import cv2
import numpy as np
import os
from glob import glob

from albumentations.pytorch import ToTensorV2
from Config import COLOR_DICT
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import Dict, List

class BinaryDataset(Dataset):
    def __init__(
        self,
        images: List[str],
        masks: List[str],
        normalization_type: int = 0,
        augmentation_transforms=None,
        color_transforms=None
    ) -> None:
        super().__init__()

        self.images = images
        self.masks = masks
        self.normalization_type = normalization_type
        self.transforms = augmentation_transforms
        self.color_transforms = color_transforms

    def load_image(self, image_path: str, binarize: bool = False) -> NDArray:
        image = cv2.imread(image_path)

        if binarize:
            image = binarize(image)

        image = image.astype(np.float32)

        if self.normalization_type == 0:
            image /= 255.0
        else:
            image = cv2.normalize(
                image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
        return image

    def encode_mask(self, y: NDArray):
        label_map = np.zeros((y.shape[0], y.shape[1], 1), dtype=np.uint8)
        label_map[np.all(y == [255, 255, 255], axis=-1)] = 1
        label_map = label_map[:, :, 0]
        return label_map

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        x = self.load_image(image_path)
        y = cv2.imread(mask_path)
        y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
        y = self.encode_mask(y)

        if self.transforms is not None:
            aug = self.transforms(image=x, mask=y)
            x = aug["image"]
            y = aug["mask"]

        if self.color_transforms is not None:
            color_aug = self.color_transforms(image=x, mask=y)
            x = color_aug["image"]

        x = np.transpose(x, axes=[2, 0, 1])   

        return x, y


class MulticlassDataset(Dataset):
    def __init__(
            self, images: List[str],
            masks: List[str],
            classes: List[str],
            normalization_type: int = 0,
            augmentation_transforms=None,
            color_transforms=None) -> None:
        
        super().__init__()

        self.images = images
        self.masks = masks
        self.base_transforms = augmentation_transforms
        self.color_transforms = color_transforms
        self.classes = classes
        self.class_indices = [x for x in range(len(self.classes))]
        self.normalization_type = normalization_type

        self.to_tensor = A.Compose(
            [
                ToTensorV2(),
            ]
        )


    def load_image(self, image_path: str, binarize: bool = False):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if binarize:
            image = binarize(image)

        image = image.astype(np.float32)

        if self.normalization_type == 0:
            image /= 255.0
        else:
            image = cv2.normalize(
                image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # do all networks require channels = 3?
        
        return image
    

    def get_color(self, key: str) -> list[int]:
        color = COLOR_DICT[key]
        color = color.split(",")
        color = [x.strip() for x in color]
        color = [int(x) for x in color]

        return color
    
    def encode_mask(self, y):
        """
        returns class-encoded map of shape HxW
        """
        label_map = np.zeros(y.shape, dtype=np.uint8)

        for idx, _class in enumerate(self.classes):
            color = self.get_color(_class)
           
            label_map[np.all(y==color, axis=-1)] = idx
        
        label_map = label_map[:,:,0]
        return label_map


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        x = self.load_image(image_path)
        y = cv2.imread(mask_path)
        y = self.encode_mask(y)

        if self.base_transforms is not None:
            base_aug = self.base_transforms(image=x, mask=y)
            x = base_aug["image"]
            y = base_aug["mask"]

        if self.color_transforms is not None:
            color_aug = self.color_transforms(image=x, mask=y)
            x = color_aug["image"]

        tensor_transf = self.to_tensor(image=x, mask=y)

        x = tensor_transf["image"]
        y = tensor_transf["mask"]
        
        return x, y


class ImageInferenceDataset(Dataset):
    def __init__(self, root_dir: str):
        self.paths = sorted(
            p for p in glob(os.path.join(root_dir, "*"))
            if p.lower().endswith((".jpg", ".png", ".jpeg"))
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # uint8, CHW, RGB
        # TODO: check if images are .tif/.tiff and use alternative loading since torchvision doesn't support tif/tiff
        img = read_image(self.paths[idx])

        meta = {
            "image_name": os.path.basename(self.paths[idx]),
            "orig_shape": (img.shape[1], img.shape[2]),  # (H, W)
        }

        return img, meta
