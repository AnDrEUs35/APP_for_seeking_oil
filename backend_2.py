from PIL import Image, ImageFilter
import numpy as np
import os
import shutil
import rasterio

class ImageMarkup:
    def __init__(self, images_path, masks_path):
        self.corner = 0
        self.images_path = images_path
        self.masks_path = masks_path

    def __crop_image(self, save_directory, mask_or_img):
        os.makedirs(f'{save_directory}/dir_{self.corner}', exist_ok=True)
        num = 0
        for i in range(0, mask_or_img.width, 1024):
            for j in range(0, mask_or_img.height, 1024):
                cropped_img = mask_or_img.crop((i, j, i+1024, j+1024))
                cropped_img.save(f'{save_directory}/dir_{self.corner}/cropped_img_{num}.tif')
                num += 1


    def __rotate_image(self, save_directory, mask_or_img):
        os.makedirs(f'{save_directory}/dir_{self.corner}', exist_ok=True)
        rotated = mask_or_img.rotate(self.corner, expand=True)
        rotated.save(f"{save_directory}/dir_{self.corner}/rot{self.corner}_img.tif")


    def work(self, save_dir):
        saved_images = os.path.join(save_dir, 'cropped_images')
        saved_masks = os.path.join(save_dir, 'cropped_masks')
        if os.path.exists(saved_images) == False:
            os.makedirs(saved_images)
        if os.path.exists(saved_masks) == False:
            os.makedirs(saved_masks)

        snaps_list = sorted(os.listdir(self.images_path))
        masks_list = sorted(os.listdir(self.masks_path))
        
        for snap, mask in zip(snaps_list, masks_list):
            img_dir = os.path.join(saved_images, snap[:-4])
            mask_dir = os.path.join(saved_masks, mask[:-4])

            if os.path.exists(img_dir) == False:
                os.makedirs(img_dir)

            if os.path.exists(mask_dir) == False:
                os.makedirs(mask_dir)
            

            with Image.open(os.path.join(self.images_path, snap)) as img, Image.open(os.path.join(self.masks_path, mask)) as mask:
                for self.corner in np.arange(0, 120, 30):
                    self.__crop_image(img_dir, img)
                    self.__rotate_image(img_dir, img)
                    self.__crop_image(mask_dir, mask)
                    self.__rotate_image(mask_dir, mask)
                self.__crop_image(mask_dir, mask)
                self.__crop_image(img_dir, img)

    def __move_excess(self, saved_images, saved_masks, moved_images, moved_masks):
        for dir_number in range(0, 120, 30):
            images_dir = f'{saved_images}/dir_{dir_number}'
            masks_dir = f'{saved_masks}/dir_{dir_number}'
            to_move = []

            for name in os.listdir(masks_dir):
                with Image.open(os.path.join(masks_dir, name)) as cr_mask:
                    arr = np.array(cr_mask)
                    if not np.any(arr > 0):
                        to_move.append(name)

            os.makedirs(os.path.join(moved_images, f'dir_{dir_number}'), exist_ok=True)
            os.makedirs(os.path.join(moved_masks, f'dir_{dir_number}'), exist_ok=True)

            for name in to_move:
                try:
                    shutil.move(os.path.join(images_dir, name), os.path.join(moved_images, f'dir_{dir_number}', name))
                    shutil.move(os.path.join(masks_dir, name), os.path.join(moved_masks, f'dir_{dir_number}', name))
                except Exception as e:
                    print(f'Ошибка при перемещении {name}: {e}')

    


class ImageChanger:


    def find_edges(self, input_path, output_path, name):
        os.makedirs(output_path, exist_ok=True)
        with Image.open(input_path) as img:
            img_gray = img.convert('L')
            edges = img_gray.filter(ImageFilter.FIND_EDGES)
            edges.save(os.path.join(output_path, name))


    def geotiff_to_tiff(self, input_path, output_path):
        os.makedirs(output_path, exist_ok=True)
        for image in os.listdir(input_path):
            with rasterio.open(os.path.join(input_path, image)) as img:
                img_array = img.read()
                bands, h, w = img_array.shape 
                if bands < 2:
                    raise ValueError(f"Недостаточно каналов: {bands}")

                r = img_array[0].astype('uint8')
                g = img_array[1].astype('uint8')
                b = np.zeros((h, w), dtype='uint8')
                rgb = np.stack([r, g, b], axis=-1)
                img_out = Image.fromarray(rgb, mode='RGB')
                img_out.save(os.path.join(output_path, image), 'TIFF')


    def sum_channels(self, input_path, output_path, band1, band2):
        os.makedirs(output_path, exist_ok=True)
        green_image = None
        NIR_image = None

        for file in os.listdir(input_path):
            if not file.lower().endswith('.tif'):
                continue
            if band1 in file:
                green_image = os.path.join(input_path, file)
            elif band2 in file:
                NIR_image = os.path.join(input_path, file)

        if green_image is None or NIR_image is None:
            raise FileNotFoundError("Не найдены файлы с каналами B3 и B5")

        with Image.open(green_image) as green, Image.open(NIR_image) as nir:
            green_arr = np.array(green, dtype='float32')
            nir_arr = np.array(nir, dtype='float32')
            eps = 1e-6
            sum_bands = (green_arr - nir_arr) / (green_arr + nir_arr + eps)
            norm_index = ((sum_bands + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(norm_index, mode='L')
            img.save(os.path.join(output_path, 'sum_result.tif'))

if __name__ == '__main__':
    a = ImageMarkup('snaps', 'masks')
    # a.work('snaps', 'masks', 'files', 'cr_masks')
    # a.geotiff_to_tiff('unready_snaps', 'ready_snaps')
    # a.sum_channels('C:/Users/Admin/Desktop/LC09_L2SP_176029_20250524_20250525_02_T1', 'be')
    # a.find_edges('be', 'ready_snaps')
