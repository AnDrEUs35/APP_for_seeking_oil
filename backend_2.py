from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import os
import rasterio
import itertools
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QComboBox, QFormLayout, QMessageBox
from osgeo import gdal
gdal.UseExceptions()



class ChannelSelectDialog(QDialog):
    def __init__(self, parent, band_count):
        super().__init__(parent)
        self.setWindowTitle("Выберите канал")
        self.combo = QComboBox(self)
        for i in range(1, band_count+1):
            self.combo.addItem(f"Канал {i}", i)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QFormLayout(self)
        layout.addRow("Канал:", self.combo)
        layout.addWidget(buttons)

    @property
    def selected_band(self):
        return self.combo.currentData()



class ImageMarkup:
    def __init__(self, images_path, masks_path):
        self.corner = 0
        self.images_path = images_path
        self.masks_path = masks_path

    def __crop_image(self, save_path, img):
        num = 0
        for i in range(0, img.width, 1024):
            for j in range(0, img.height, 1024):
                cropped = img.crop((i, j, i + 1024, j + 1024))
                cropped.save(f'{save_path}_cropped_{num}.tif')
                num += 1


    def work(self, save_dir):
        saved_images = os.path.join(save_dir, 'cropped_images')
        saved_masks = os.path.join(save_dir, 'cropped_masks')
        os.makedirs(saved_images, exist_ok=True)
        os.makedirs(saved_masks, exist_ok=True)

        snaps_list = sorted(os.listdir(self.images_path))
        
        if not self.masks_path:
            empty_masks_path = os.path.join(save_dir, 'original_empty_masks')
            os.makedirs(empty_masks_path, exist_ok=True)
            for snap in snaps_list:
                with Image.open(os.path.join(self.images_path, snap)) as img:
                    empty_mask = Image.new('L', img.size, color=0)
                    empty_mask.save(os.path.join(empty_masks_path, f'{snap}'))
                    self.masks_path = empty_masks_path

        masks_list = sorted(os.listdir(self.masks_path))
        for snap, mask in zip(snaps_list, masks_list):
            img_path = os.path.join(self.images_path, snap)
            mask_path = os.path.join(self.masks_path, mask)

            with Image.open(img_path) as img, Image.open(mask_path) as msk:
                for angle in (0, 30, 60, 90):
                    rotated_img = img.rotate(angle)
                    rotated_msk = msk.rotate(angle)

                    img_path = os.path.join(saved_images, f'{snap[:-4]}_angle{angle}')
                    msk_path = os.path.join(saved_masks, f'mask_{mask[:-4]}_angle{angle}')

                    self.__crop_image(img_path, rotated_img)
                    self.__crop_image(msk_path, rotated_msk)


    def tiff_to_png(self, dir_path, out_path):
        for f in os.listdir(dir_path):
            in_path = dir_path+f
            out_path = out_path + os.path.splitext(f)[0]+'.png'
            ds = gdal.Translate(out_path, in_path, options="-scale -ot Byte")
        


    def delete_excess(self, images_path, masks_path):
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
                if img[:5] != 'mask_':
                    return Exception('Маска не найдена или названа некорректно (должно начинаться с "mask_").')
                mask_name = 'mask_' + img
                path = os.path.join(masks_path, mask_name)
                os.remove(path)

    


class ImageChanger:
        

    def find_edges(self, input_path, output_path, name):
        os.makedirs(output_path, exist_ok=True)
        with Image.open(input_path) as img:
            img_gray = img.convert('L')
            edges = img_gray.filter(ImageFilter.FIND_EDGES)
            edges = edges.filter(ImageFilter.EDGE_ENHANCE)
            edges.save(os.path.join(output_path, name))


    def geotiff_to_tiff(self, input_path, output_path):
        os.makedirs(output_path, exist_ok=True)
        formats = [
            "uint8",
            "uint16",
            "uint32",
            "int16",
            "int32",
            "float32"]
        
        for image_name in os.listdir(input_path):
            # Пропускаем системные файлы и файлы, не являющиеся TIFF
            if image_name.startswith('.') or not image_name.lower().endswith(('.tif', '.tiff')):
                continue

            source_path = os.path.join(input_path, image_name)
            output_filename = f'{image_name}'
            destination_path = os.path.join(output_path, output_filename)

            try:
                with rasterio.open(source_path) as src:
                    band_count = src.count
                if band_count > 1:
                    # спрашиваем пользователя, какой канал читать
                    dlg = ChannelSelectDialog(self, band_count)
                    if dlg.exec() != QDialog.DialogCode.Accepted:
                        return False
                    band = dlg.selected_band
                else:
                    band = 1
                    img_array = src.read(band)
                        
                    # Получаем метаданные (профиль) исходного файла
                    profile = src.profile
                
                # Обновляем профиль для нашего выходного файла:
                # 1. Устанавливаем тип данных float32
                # 2. Указываем, что у нас будет только 1 канал
                print(profile['dtype'])
                if profile['dtype'] == 'float32':
                    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
                    # Создаем новый файл TIFF в режиме записи с обновленным профилем
                    with rasterio.open(destination_path, 'w', **profile) as dst:
                        dst.write(img_array.astype(rasterio.float32), 1)

                elif profile['dtype'] == 'uint32':
                    profile.update(dtype=rasterio.uint32, count=1, compress='lzw')
                    with rasterio.open(destination_path, 'w', **profile) as dst:
                        dst.write(img_array.astype(rasterio.float32), 1)

                elif profile['dtype'] == 'uint16':
                    profile.update(dtype=rasterio.uint16, count=1, compress='lzw')
                    with rasterio.open(destination_path, 'w', **profile) as dst:
                        dst.write(img_array.astype(rasterio.uint16), 1)

                elif profile['dtype'] == 'uint8':
                    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
                    with rasterio.open(destination_path, 'w', **profile) as dst:
                        dst.write(img_array.astype(rasterio.uint16), 1)
                else:
                    QMessageBox.warning(f'Предупреждение: неподдерживаемый формат данных ({profile['dtype']})')

                print(f"Файл '{image_name}' успешно преобразован в '{output_filename}'")

            except Exception as e:
                print(f"⚠️ Ошибка при обработке файла {image_name}: {e}")


    def sum_channels(self, output_path, bands):
        if len(bands) == 0:
            raise FileNotFoundError("Не найдены файлы с выбранными каналами")
        eps = 1e-6
        if len(bands) == 2:
            with Image.open(bands[0]) as band1, Image.open(bands[1]) as band2:
                band1_arr = np.array(band1, dtype='float32')
                band2_arr = np.array(band2, dtype='float32')
                sum_bands = (band1_arr - band2_arr) / (band1_arr + band2_arr + eps)
        elif len(bands) == 4:
            with Image.open(bands[0]) as band1, Image.open(bands[1]) as band2, Image.open(bands[2]) as band3, Image.open(bands[3]) as band4:
                band1_arr = np.array(band1, dtype='float32')
                band2_arr = np.array(band2, dtype='float32')
                band3_arr = np.array(band3, dtype='float32')
                band4_arr = np.array(band4, dtype='float32')
                sum_bands = (band1_arr + band2_arr) / (band3_arr + band4_arr + eps)
        else:
            return 'Несоответствие количества файлов'

        norm_index = ((sum_bands + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(norm_index, mode='L')
        img.save(os.path.join(output_path, 'sum_result.tif'))

    def create_mask_from_contours(self, image_width: int, image_height: int, contours: list[list[tuple]]):
        """
        Создает бинарную маску из списка контуров.
        
        Аргументы:
            image_width (int): Ширина исходного изображения.
            image_height (int): Высота исходного изображения.
            contours (list[list[tuple]]): Список контуров, где каждый контур - это список
                                          кортежей (x, y) в координатах исходного изображения.
                                          Предполагается, что контуры являются замкнутыми полигонами.
        
        Returns:
            PIL.Image.Image: Бинарное изображение-маска (режим 'L'), где контуры белые (255), фон черный (0).
        """
        mask_image = Image.new('L', (image_width, image_height), 0) # Создаем белое изображение
        draw = ImageDraw.Draw(mask_image) # Отрисовка маски
        
        for contour in contours:
            # Преобразуем список кортежей в плоский список координат для ImageDraw.polygon
                # point — это кортеж (x, y)
            flat_contour = list(itertools.chain.from_iterable(contour)) # chain позволяет "пройтись" по всем элементам из всех переданных списков, как будто это один список.
            if len(flat_contour) >= 6: # Полигон требует как минимум 3 точки (6 координат)
                draw.polygon(flat_contour, fill=255) # Заполняем чёрным цветом
        
        return mask_image
    

class NeuroBackEnd:


    def overlay_mask(self, image_path, mask_to_overlay_path, output_path):
        with Image.open(image_path).convert('RGB') as img, Image.open(mask_to_overlay_path).convert('L') as mask:
            img_arr = np.array(img)
            mask_arr = np.array(mask)

            overlay = img_arr.copy()
            overlay[mask_arr > 0] = (255, 0, 0)

            overlay_img = Image.fromarray(overlay).convert('RGB')
            overlay_path = os.path.join(output_path, f'overlayed_{os.path.basename(image_path)}')
            overlay_img.save(overlay_path, 'TIFF')
            return overlay_path
            

if __name__ == '__main__':
   pass
    