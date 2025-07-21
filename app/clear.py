import os
import numpy as np
from PIL import Image
masks_path = 'C:/Users/Admin/Desktop/cropped_masks_png/'
images_path = 'C:/Users/Admin/Desktop/cropped_images_png/'
deleted_images = []
for img in os.listdir(images_path):
    path = os.path.join(images_path, img)
    with Image.open(path) as image:
        img_arr = np.array(image)
        non_zero_pixels = np.count_nonzero(img_arr)
        if non_zero_pixels == 0:
            deleted_images.append(img)
            os.remove(path)
if deleted_images:
    for img in deleted_images:
        mask_name = 'mask_' + img
        path = os.path.join(masks_path, mask_name)
        os.remove(path)
