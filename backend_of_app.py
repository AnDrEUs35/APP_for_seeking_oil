from PIL import Image, ImageFilter
import numpy as np
import os
import shutil
import rasterio

class Image_markup:
    def __init__(self, images_path, masks_path):
        self.corner = 0
        self.images_path = images_path
        self.masks_path = masks_path


    def __crop_image(self, save_directory, mask_or_img):
        os.makedirs(f'{save_directory}/dir_{self.corner}', exist_ok=True)
        num = 0
        for i in np.arange(0, mask_or_img.width + 1024, 1024):
            for j in np.arange(0, mask_or_img.height + 1024, 1024):
                if i > mask_or_img.width or j > mask_or_img.height:
                    break
                num += 1
                cropped_img = mask_or_img.crop((i, j, i+1024, j+1024))
                cropped_img.save(f'{save_directory}/dir_{self.corner}/cr_img_{num}.tif')


    def __clear_all(self, delete_dirs):
        for corner in np.arange(0, 120, 30):
            try:
                shutil.rmtree(f"{delete_dirs[0]}/dir_{corner}")
                shutil.rmtree(f"{delete_dirs[1]}/dir_{corner}")
            except Exception:
                pass
    

    def __rotate_image(self, save_directory, corner, mask_or_img):
        self.corner = corner
        os.makedirs(f'{save_directory}/dir_{self.corner}', exist_ok=True)
        mask_or_img = mask_or_img.rotate(corner, expand=True)
        mask_or_img.save(f"{save_directory}/dir_{self.corner}/rot{self.corner}_img.tif")


    def work(self, path_to_images, path_to_masks, save_dir):
        saved_images = 'cropped_images'
        saved_masks = 'cropped_masks'
        os.makedirs(os.path.join(save_dir, saved_images))
        os.makedirs(os.path.join(save_dir, saved_masks))
        for snap, mask in os.listdir(self.images_path), os.listdir(self.masks_path):
            with Image.open(os.path.join(path_to_images, snap)) as img, Image.open(os.path.join(path_to_masks, mask)) as mask:
                for corner in np.arange(0, 120, 30):
                    self.__crop_image(saved_images, img)
                    self.__rotate_image(saved_images, corner, img)
                    self.__crop_image(saved_masks, mask)
                    self.__rotate_image(saved_masks, corner, mask)
                self.__crop_image(saved_masks, mask)
                self.__crop_image(saved_images, img)


    def __move_excess(self, saved_images, saved_masks, moved_images, moved_masks):
        for dir_number in np.arange(0, 120, 30):
            to_move = []
            for name in os.listdir(f'{saved_images}'):
                with Image.open(f'{saved_masks}/dir_{dir_number}/{name}') as cr_mask:
                    arr = np.array(cr_mask)
                    if not np.any(arr > 0):
                        to_move.append(name)
            for name in to_move:
                try:
                    shutil.move(os.path.join(saved_images, name), os.path.join(moved_images, name))
                    shutil.move(os.path.join(saved_masks, name), os.path.join(moved_masks, name))
                except Exception as e:
                    print(f'Ошибка при удалении {name}: {e}')


    def find_edges(self, input_path, output_path):
        for image in os.listdir(input_path):
            with Image.open(f'{input_path}/{image}') as img:
                img_gray = img.convert('L')
                edges = img_gray.filter(ImageFilter.FIND_EDGES)
                edges.save(f'{output_path}/{image}')


    def geotiff_to_tiff(self, input_path, output_path):
        for image in os.listdir(input_path):
            with rasterio.open(f'{input_path}/{image}') as img:
                img_array = img.read()
                print(img_array)
                
                bands, h, w = img_array.shape 
                if bands != 2:
                    raise ValueError(f"Неожиданное число каналов: {bands}")
                
                r = img_array[0].astype('uint8')
                g = img_array[1].astype('uint8')  

                b = np.zeros((h, w), dtype='uint8')
                rgb = np.stack([r, g, b], axis=-1)
                
                img = Image.fromarray(rgb, mode='RGB')
                img.save(f'{output_path}/{image}', 'TIFF')
                print('Изображение преобразовано.')
            

    def sum_channels(self, input_path, output_path):
        for file in os.listdir(input_path):
            if not file.lower().endswith('.tif'):
                os.remove(f'{input_path}/{file}')
            if 'B3' in file:
                green_image = os.path.join(input_path, file)
            elif 'B5' in file:
                NIR_image = os.path.join(input_path, file)
        with Image.open(green_image) as green, Image.open(NIR_image) as nir:
            green_arr, nir_arr = np.array(green), np.array(nir)
            eps = 1e-6
            sum_bands = (green_arr - nir_arr) / (green_arr + nir_arr + eps)

            norm_index = ((sum_bands + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

            img = Image.fromarray(norm_index, mode='L')
            img.save(f'{output_path}/what_do_i_do_here.tif')
            


if __name__ == '__main__':
    a = Image_markup('snaps', 'masks')
    # a.work()
    # a.geotiff_to_tiff('unready_snaps', 'ready_snaps')
    # a.sum_channels('C:/Users/Admin/Desktop/LC09_L2SP_176029_20250524_20250525_02_T1', 'be')
    # a.find_edges('be', 'ready_snaps')

