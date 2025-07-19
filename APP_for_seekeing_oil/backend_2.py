from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import os
import rasterio
import itertools
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QComboBox, QFormLayout, QWidget
from osgeo import gdal
import torch
from ..network.model import Model
from torchvision import transforms
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


    def __crop_image(self, save_path, img):
        num = 0
        for i in range(0, img.width, self.size):
            for j in range(0, img.height, self.size):
                cropped = img.crop((i, j, i + self.size, j + self.size))
                cropped.save(f'{save_path}_cropped_{num}.tif')
                num += 1


    def work(self, images_path, masks_path, save_dir):
        self.size = 1024
        saved_images = os.path.join(save_dir, 'cropped_images')
        saved_masks = os.path.join(save_dir, 'cropped_masks')
        os.makedirs(saved_images, exist_ok=True)
        os.makedirs(saved_masks, exist_ok=True)

        snaps_list = sorted(os.listdir(images_path))
        
        if not masks_path:
            empty_masks_path = os.path.join(save_dir, 'original_empty_masks')
            os.makedirs(empty_masks_path, exist_ok=True)
            for snap in snaps_list:
                with Image.open(os.path.join(images_path, snap)) as img:
                    empty_mask = Image.new('L', img.size, color=0)
                    empty_mask.save(os.path.join(empty_masks_path, f'{snap}'))
                    masks_path = empty_masks_path

        masks_list = sorted(os.listdir(masks_path))
        for snap, mask in zip(snaps_list, masks_list):
            img_path = os.path.join(images_path, snap)
            mask_path = os.path.join(masks_path, mask)

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
            in_path = os.path.join(dir_path, f)
            output_path = out_path + '/' + os.path.splitext(f)[0]+'.png'
            ds = gdal.Translate(output_path, in_path, options="-scale -ot Byte")
        


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


    def geotiff_to_tiff(self, input_path, output_path, parent):
        os.makedirs(output_path, exist_ok=True)
        # спрашиваем пользователя, какой канал читать
        count_bands = []
        for img in os.listdir(input_path):
            with rasterio.open(os.path.join(input_path, img)) as test:
                if test.count > 1:
                    count_bands.append(test.count)
        if len(set(count_bands)) <= 1:
            dlg = ChannelSelectDialog(parent, count_bands[0])
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return False
        else:
            raise Exception('В выборке не все изображения содержат одинаковое количество каналов > 1')
        for image_name in os.listdir(input_path):
            # Пропускаем системные файлы и файлы, не являющиеся TIFF
            if image_name.startswith('.') or not image_name.lower().endswith(('.tif', '.tiff')):
                continue

            source_path = os.path.join(input_path, image_name)
            output_filename = f'{image_name}'
            destination_path = os.path.join(output_path, output_filename)

            with rasterio.open(source_path) as src:
                band_count = src.count
                if band_count > 1:
                    band = dlg.selected_band
                if band_count != count_bands[0]:
                    dlg = ChannelSelectDialog(parent, band_count)
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

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = Model("Unet", "resnet34", in_channels=3, out_classes=1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict_mask(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        tensor = self._preprocess(image)
        with torch.no_grad():
            output = self.model(tensor)
            mask = torch.sigmoid(output.squeeze(0)).squeeze().cpu().numpy()
        return (mask > 0.5).astype(np.uint8)
    
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # можно заменить на реальный размер, если известен
            transforms.ToTensor()
        ])
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor
            

if __name__ == '__main__':
   pass
    