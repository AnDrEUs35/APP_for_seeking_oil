from torch.utils.data import Dataset as BaseDataset
import torch
import numpy as np
import os
import cv2


class Dataset(BaseDataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        input_image_reshape,
        foreground_class,
        augmentation=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_filepaths = [
            os.path.join(images_dir, image_id) for image_id in self.ids
        ]
        self.masks_filepaths = [
            os.path.join(masks_dir, image_id) for image_id in self.ids
        ]

        self.input_image_reshape = input_image_reshape
        self.foreground_class = foreground_class
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.imread(
            self.images_filepaths[i], cv2.IMREAD_COLOR_RGB
        )
        image = cv2.resize(image, self.input_image_reshape)

        mask = cv2.imread(self.masks_filepaths[i], 0)
        mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)

        mask_remap = cv2.resize(mask_remap, self.input_image_reshape)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]
    
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0

        mask_remap = torch.tensor(mask_remap).long()

        return image, mask_remap

    def __len__(self):
        return len(self.ids)