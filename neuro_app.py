import sys
from PyQt6.QtWidgets import (QApplication, QLabel, QHBoxLayout, QMainWindow, QPushButton, QVBoxLayout, QWidget,
                             QTreeView, QFileDialog, QInputDialog, QComboBox, QMessageBox)
from PyQt6.QtCore import Qt, QPoint, QModelIndex, QRect
from PyQt6.QtGui import QFileSystemModel, QPainter, QPixmap, QMouseEvent, QWheelEvent, QPen, QColor, QBrush, QKeySequence, QShortcut


from backend_2 import *

class PaintWidget(QWidget):
    

    def __init__(self):
        super().__init__()
        self.pixmap = QPixmap()
        self.scale_factor = 1.0  # Текущий масштаб изображения
        self.pan_start_position = QPoint() # Начальная точка для панорамирования
        self.pixmap_offset = QPoint() # Смещение изображения относительно центра виджета
        self.is_panning = False # Флаг, указывающий, происходит ли сейчас панорамирование

        # Для рисования контуров
        self.drawing_enabled = False # Включен ли режим рисования
        self.drawing_mode = 'points' # 'points' (по точкам) или 'freehand' (свободное рисование)
        self.is_drawing = False # Рисуется ли сейчас контур (т.е. добавляются ли точки для текущего контура)
        self.current_contour = [] # Точки текущего рисуемого контура (список QPoint)
        self.all_contours = [] # Все завершенные контуры (список списков QPoint)

        # Устанавливаем минимальный размер, чтобы виджет не был слишком маленьким
        self.setMinimumSize(600, 600)

        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.remove_last_contour)

    def load_image(self, path):
        if self.pixmap.load(path) == True:
            # Сбрасываем масштаб и смещение при загрузке нового изображения
            self.scale_factor = 1.0
            self.pixmap_offset = QPoint() # Центрируем изображение
            self.clear_contours() # Очищаем контуры при загрузке нового изображения
            self.update()  # Запрашиваем перерисовку виджета
            return True
        else:
            print(f"Ошибка: не удалось загрузить изображение по пути {path}")
            return False

    def set_drawing_enabled(self, enabled: bool):
        self.drawing_enabled = enabled
        if enabled == False:
            self.is_drawing = False # Останавливаем рисование, если режим рисования выключен
            self.current_contour = [] # Очищаем текущий контур при выключении режима
        self.update() # Обновляем виджет, чтобы отразить изменение режима

    def set_drawing_mode(self, mode: str):
        if mode in ['points', 'freehand']:
            self.drawing_mode = mode
            self.is_drawing = False # Сбрасываем состояние рисования
            self.current_contour = [] # Очищаем текущий контур при смене режима
            self.update()
        else:
            print(f"Неизвестный режим рисования: {mode}")

    def clear_contours(self):
        self.current_contour = []
        self.all_contours = []
        self.update()

    def get_contours_in_image_coords(self):
        """
        Возвращает список всех нарисованных контуров, преобразованных в
        координаты исходного изображения.
        """
        if self.pixmap.isNull() == True:
            return []

        # Вычисляем размеры и положение изображения с учетом масштаба
        target_width = int(self.pixmap.width() * self.scale_factor) # Отображаемые размеры изображения
        target_height = int(self.pixmap.height() * self.scale_factor)
        widget_center = self.rect().center() #
        pixmap_top_left = widget_center - QPoint(target_width // 2, target_height // 2) + self.pixmap_offset

        transformed_contours = []
        # Обрабатываем завершенные контуры, преобразуя их в один
        for contour in self.all_contours:
            transformed_single_contour = []
            for qpoint in contour:
                # Преобразование координат виджета в координаты изображения
                x_image = (qpoint.x() - pixmap_top_left.x()) / self.scale_factor
                y_image = (qpoint.y() - pixmap_top_left.y()) / self.scale_factor
                transformed_single_contour.append((int(x_image), int(y_image)))
            transformed_contours.append(transformed_single_contour)
        
        return transformed_contours


    def wheelEvent(self, event: QWheelEvent):
        """Обрабатывает событие прокрутки колеса мыши для масштабирования."""
        # Отключаем масштабирование во время рисования для предотвращения искажений
        if self.drawing_enabled == True: # Масштабирование не работает, если режим рисования включен
            return

        angle = event.angleDelta().y()
        if angle > 0:
            factor = 1.1 
        else:
            factor = 1 / 1.1
        
        self.scale_factor *= factor
        self.update()  # Перерисовываем виджет с новым масштабом


    def remove_last_contour(self):
        if self.all_contours:
            self.all_contours.pop()
            self.update()
        else:
            QMessageBox.information(self, 'Удаление последнего контура', 'На изображении ещё нет контуров.')


    def mousePressEvent(self, event: QMouseEvent):
        """Обрабатывает нажатие кнопки мыши для начала панорамирования или добавления точки/завершения контура."""
        if self.drawing_enabled == True:
            if event.button() == Qt.MouseButton.LeftButton:
                self.is_drawing = True # Указываем, что начали рисовать (добавлять точки)
                if self.drawing_mode == 'points':
                    self.current_contour.append(event.pos()) # Добавляем точку к текущему контуру
                elif self.drawing_mode == 'freehand':
                    self.current_contour = [event.pos()] # Начинаем новый контур для свободного рисования
                self.update()
            elif event.button() == Qt.MouseButton.RightButton:
                # Завершаем текущий контур по правому клику (актуально для режима 'points')
                if self.drawing_mode == 'points' and self.is_drawing and len(self.current_contour) > 1:
                    # Замыкаем контур, добавляя первую точку в конец
                    self.current_contour.append(self.current_contour[0]) 
                    self.all_contours.append(list(self.current_contour)) # Добавляем копию списка
                self.current_contour = [] # Сбрасываем текущий контур
                self.is_drawing = False # Завершаем режим добавления точек
                self.update()
        else: # Панорамирование, если режим рисования выключен
            if event.button() == Qt.MouseButton.LeftButton:
                self.is_panning = True
                self.pan_start_position = event.pos()


    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing_enabled and self.drawing_mode == 'freehand' and self.is_drawing == True:
            self.current_contour.append(event.pos())
            self.update()
        elif self.is_panning:
            delta = event.pos() - self.pan_start_position
            self.pixmap_offset += delta
            self.pan_start_position = event.pos()
            self.update()


    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.drawing_enabled and self.drawing_mode == 'freehand' and self.is_drawing:
            if event.button() == Qt.MouseButton.LeftButton:
                if len(self.current_contour) > 1:
                    # Замыкаем контур, добавляя первую точку в конец
                    self.current_contour.append(self.current_contour[0])
                    self.all_contours.append(list(self.current_contour))
                self.current_contour = []
                self.is_drawing = False
                self.update()
        elif self.is_panning:
            if event.button() == Qt.MouseButton.LeftButton:
                self.is_panning = False


    def paintEvent(self, event):
        super().paintEvent(event) # ВЫзов родительского метода paintEvent Мы его переопределяем
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform) # Включает сглаживание при масштабировании изображений для лучшего качества.

        if self.pixmap.isNull() == False:  # Проверяем, есть ли изображение.
            # Вычисляем размеры и положение изображения с учетом масштаба
            # Пока плохо понимаю
            target_width = int(self.pixmap.width() * self.scale_factor)
            target_height = int(self.pixmap.height() * self.scale_factor)
            widget_center = self.rect().center()
            pixmap_top_left = widget_center - QPoint(target_width // 2, target_height // 2) + self.pixmap_offset
            target_rect = QRect(pixmap_top_left.x(), pixmap_top_left.y(), target_width, target_height)

            # Рисуем изображение
            painter.drawPixmap(target_rect, self.pixmap)

        # Настройка пера для рисования контуров
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen) # Красный цвет для обводки, толщина линии

        # Настройка кисти для закрашивания контуров (полупрозрачный красный)
        brush = QBrush(QColor(255, 0, 0, 100)) # Красный с 100 alpha (полупрозрачный)
        painter.setBrush(brush)  

        # Рисуем все завершенные контуры
        for contour in self.all_contours:
            if len(contour) > 1:
                # drawPolygon автоматически замыкает контур и закрашивает его
                painter.drawPolygon(contour) 

        # Рисуем текущий (незавершенный) контур, если он рисуется
        if self.is_drawing and len(self.current_contour) > 1:
            # Для текущего контура также используем drawPolygon для закрашивания
            painter.drawPolygon(self.current_contour)
            # Дополнительно можно нарисовать точки, чтобы было видно, где пользователь кликнул (только для режима 'points')
            if self.drawing_mode == 'points':
                # Отключаем кисть для рисования точек, чтобы они не были закрашены
                painter.setBrush(Qt.BrushStyle.NoBrush) 
                for point in self.current_contour:
                    painter.drawEllipse(point, 3, 3) # Нарисовать маленький кружок вокруг каждой точки
                # Возвращаем кисть для следующего рисования контуров
                painter.setBrush(brush)













class OilApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dir_path = None
        
        self.setWindowTitle('Идентификация нефтепродуктов на спутниковых снимках')
        self.image_changer = ImageChanger()
        self.current_image_path = None # Хранит путь к текущему загруженному изображению
        self.initUI()


    def initUI(self):
        central = QWidget()
        main_lay = QVBoxLayout()
        layV1 = QVBoxLayout() # Вертикальный слой для кнопок и дерева файлов
        layH = QHBoxLayout() # Горизонтальный слой для дерева и области рисования

        self.label = QLabel('Выберите каталог с изображениями')
        self.label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        self.dir_button = QPushButton("Выбрать папку")
        self.dir_button.clicked.connect(self.choose_directory)

        # Используем наш новый виджет для рисования
        self.paint_widget = PaintWidget()
        
        # Модель файловой системы и дерево для отображения
        self.model = QFileSystemModel()
        self.model.setRootPath('')  # Изначально корень пустой
        
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setFixedHeight(340)
        self.tree.clicked.connect(self.on_file_clicked)
        self.tree.hide() # Скрываем дерево, пока не выбрана папка
        # Скрываем лишние колонки (размер, тип, дата)
        for col in range(1, self.model.columnCount()):
            self.tree.hideColumn(col)

        self.instrument_label = QLabel('Инструменты для преобразования выбранных изображений')

        self.clear_button = QPushButton('Удалить всё из директории')
        self.clear_button.clicked.connect(self.clear_directory)

        self.geotiff_to_tiff_button = QPushButton('Преобразовать GeoTiff в Tiff')
        self.geotiff_to_tiff_button.clicked.connect(self.geotiff_to_tiff)

        self.find_edges_button = QPushButton('Найти границы на изображении')
        self.find_edges_button.clicked.connect(self.find_edges)

        self.crop_label = QLabel('Подготовка данных для обучения модели')

        self.sum_channels_button = QPushButton('Создание синтетического изображения')
        self.sum_channels_button.clicked.connect(self.sum_channels)
        self.sum_channels_button.setToolTip('Выберите формулу выше и в появившемся окне укажите пути до нужных каналов одного изображения.')

        self.formula_list = ['NDOI = ρλ(green)−ρλ(NIR) / ρλ(green)+ρλ(NIR)',
                             'G‑SWIR = ρλ(green) −ρλ(SWIR2) / ρλ(green)+ρλ(SWIR2)',
                             'CaBGS = ρλ(coastal_aerosol)+ρλ(blue) / ρλ(green)+ρλ(SWIR2)']
        self.formula_box = QComboBox(self)
        self.formula_box.addItems(self.formula_list)
        self.formula_box.setToolTip('')

        self.crop_button = QPushButton('Подготовить выборку')
        self.crop_button.clicked.connect(self.run_markup)
        self.crop_button.setToolTip('Вы должны будете выбрать три папки: со снимками, с масками, путь сохранения.' \
        ' \n В итоге получите разбитые и отсортированные изображения \n для обучающей выборки.')

        # Новые элементы для выделения контуров
        self.drawing_label = QLabel('Инструменты для выделения контуров')
        self.toggle_drawing_button = QPushButton('Включить/Выключить рисование')
        self.toggle_drawing_button.setCheckable(True) # Сделать кнопку переключаемой
        self.toggle_drawing_button.clicked.connect(self.toggle_drawing_mode)
        
        self.clear_contours_button = QPushButton('Очистить контуры')
        self.clear_contours_button.clicked.connect(self.paint_widget.clear_contours)

        self.save_mask_button = QPushButton('Сохранить маску контуров')
        self.save_mask_button.clicked.connect(self.save_contours_as_mask)

        # Выбор режима рисования
        self.drawing_mode_label = QLabel('Режим рисования:')
        self.drawing_mode_combo = QComboBox(self)
        self.drawing_mode_combo.addItems(['По точкам', 'Свободное рисование'])
        self.drawing_mode_combo.currentIndexChanged.connect(self.set_current_drawing_mode)

        layH_drawing_mode = QHBoxLayout()
        layH_drawing_mode.addWidget(self.drawing_mode_label)
        layH_drawing_mode.addWidget(self.drawing_mode_combo)


        # Собираем левую часть интерфейса
        layV1.addWidget(self.dir_button)
        layV1.addWidget(self.tree)
        layV1.addWidget(self.instrument_label)
        layV1.addWidget(self.geotiff_to_tiff_button)
        layV1.addWidget(self.clear_button)
        layV1.addWidget(self.find_edges_button)
        layV1.addWidget(self.formula_box)
        layV1.addWidget(self.sum_channels_button)
        layV1.addWidget(self.crop_label)
        layV1.addWidget(self.crop_button)
        layV1.addWidget(self.drawing_label) # Добавляем новый раздел для рисования
        layV1.addLayout(layH_drawing_mode) # Добавляем выбор режима рисования
        layV1.addWidget(self.toggle_drawing_button)
        layV1.addWidget(self.clear_contours_button)
        layV1.addWidget(self.save_mask_button)


        # Собираем основной горизонтальный слой
        layH.addLayout(layV1)
        layH.addWidget(self.paint_widget, 1) # Добавляем виджет рисования с растяжением

        # Собираем главный слой окна
        main_lay.addWidget(self.label)
        main_lay.addLayout(layH)
    
        central.setLayout(main_lay)
        self.setCentralWidget(central)


    def choose_directory(self):
        self.dir_path = QFileDialog.getExistingDirectory(
            self, "Выберите директорию", "")
        if self.dir_path:
            self.label.setText(f"Текущая папка: {self.dir_path}")
            self.model.setRootPath(self.dir_path)
            self.tree.setRootIndex(self.model.index(self.dir_path))
            self.tree.show()


    def geotiff_to_tiff(self):
        """Преобразует GeoTiff файлы в Tiff."""
        input_path = QFileDialog.getExistingDirectory(self, 'Укажите путь к папке с файлами GeoTiff')
        if not input_path: return
        output_path = QFileDialog.getExistingDirectory(self, 'Укажите путь, куда файл нужно сохранить')
        if not output_path: return
        try:
            self.image_changer.geotiff_to_tiff(input_path, output_path)
            QMessageBox.information(self, "Успех", "Преобразование GeoTiff в Tiff завершено.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при преобразовании: {e}")
            print(e)


    def find_edges(self):
        index = self.tree.currentIndex()
        if not index.isValid() or self.model.isDir(index) == True:
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, выберите файл изображения.")
            return
        input_path = self.model.filePath(index)
        output_path = QFileDialog.getExistingDirectory(self, 'Укажите путь, куда файл нужно сохранить')
        if not output_path: return
        try:
            file_name = os.path.basename(input_path)
            name_without_ext = os.path.splitext(file_name)[0]
            output_file_name = f"{name_without_ext}_edges.tif"
            self.image_changer.find_edges(input_path, output_path, output_file_name)
            QMessageBox.information(self, "Успех", f"Границы найдены и сохранены в {os.path.join(output_path, output_file_name)}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при поиске границ: {e}")

    
    def sum_channels(self):
        input_path = self.dir_path
        if not input_path:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите директорию с исходными изображениями.")
            return
        try:
            formula = self.formula_box.currentText()
            if formula == self.formula_list[0]:
                band1, band2 = QFileDialog.getOpenFileNames(self, "Выберите 2 файла в порядке: green, NIR", input_path, "TIFF файлы (*.tif *.tiff);;Все файлы (*)")
                output_path = QFileDialog.getExistingDirectory(self, 'Укажите путь, куда сохранить синтетическое изображение')
                if not output_path: return
                self.image_changer.sum_channels(output_path, band1, band2)
            elif formula == self.formula_list[1]:
                band1, band2 = QFileDialog.getOpenFileNames(self, "Выберите 2 файла в порядке: green, SWIR-2", input_path, "TIFF файлы (*.tif *.tiff);;Все файлы (*)")
                output_path = QFileDialog.getExistingDirectory(self, 'Укажите путь, куда сохранить синтетическое изображение')
                if not output_path: return
                self.image_changer.sum_channels(output_path, band1, band2)
            elif formula == self.formula_list[2]:
                band1, band2, band3, band4 = QFileDialog.getOpenFileNames(self, "Выберите 4 файла в порядке: coastal_aerosol, blue, green, SWIR-2", input_path, "TIFF файлы (*.tif *.tiff);;Все файлы (*)")
                output_path = QFileDialog.getExistingDirectory(self, 'Укажите путь, куда сохранить синтетическое изображение')
                if not output_path: return
                self.image_changer.sum_channels(output_path, band1, band2, band3, band4)
            QMessageBox.information(self, "Успех", "Сложение каналов завершено.")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Ошибка", f"Не найдены необходимые файлы каналов: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при сложении каналов: {e}")
        


    def clear_directory(self):
        directory = QFileDialog.getExistingDirectory(self, 'Выбор директории, которую нужно полностью очистить')
        if not directory: return
        reply = QMessageBox.question(self, 'Подтверждение очистки',
                                     f"Вы уверены, что хотите полностью очистить директорию '{directory}'? Это действие необратимо.",
                                     QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isfile(item_path) == True:
                        os.remove(item_path)
                    elif os.path.isdir(item_path) == True:
                        shutil.rmtree(item_path)
                QMessageBox.information(self, "Успех", f"Директория '{directory}' успешно очищена.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось очистить директорию: {e}")


    def on_file_clicked(self, index: QModelIndex):
        file_path = self.model.filePath(index)
        if self.model.isDir(index) == False: # Проверяем, что это файл, а не папка
            if self.paint_widget.load_image(file_path):
                self.current_image_path = file_path # Сохраняем путь к загруженному изображению
        else:
            self.current_image_path = None # Очищаем путь, если выбрана директория


    def run_markup(self):
        snaps_path = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if snaps_path == None:
            return
        masks_path = QFileDialog.getExistingDirectory(self, "Выберите папку с масками")
        if masks_path == None:
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if save_dir == None:
            return

        self.markup = ImageMarkup(snaps_path, masks_path)
        try:
            self.markup.work(save_dir)
            QMessageBox.information(self, "Успех", "Подготовка выборки завершена.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при подготовке выборки: {str(e)}")


    def toggle_drawing_mode(self):
        '''Включаем или выключаем режим рисования'''
        is_checked = self.toggle_drawing_button.isChecked()
        self.paint_widget.set_drawing_enabled(is_checked)
        
        # Обновляем метку с инструкциями в зависимости от включенного состояния и текущего режима рисования
        if is_checked:
            current_mode = self.paint_widget.drawing_mode
            if current_mode == 'points':
                self.label.setText("Режим рисования включен: По точкам. ЛКМ для добавления точки, ПКМ для завершения контура.")
            else: # freehand
                self.label.setText("Режим рисования включен: Свободное рисование. Удерживайте ЛКМ для рисования, отпустите для завершения контура.")
        else:
            self.label.setText("Режим рисования выключен. Используйте колесо мыши для масштабирования, ЛКМ для панорамирования.")

    def set_current_drawing_mode(self, index: int):
        """Устанавливает режим рисования в PaintWidget на основе выбора QComboBox."""
        mode_map = {
            0: 'points',
            1: 'freehand'
        }
        selected_mode = mode_map.get(index, 'points') # получаем выбранный пользователем режим
        self.paint_widget.set_drawing_mode(selected_mode)
        
        # Обновляем метку с инструкциями после смены режима
        if self.paint_widget.drawing_enabled:
            if selected_mode == 'points':
                self.label.setText("Режим рисования включен: По точкам. ЛКМ для добавления точки, ПКМ для завершения контура.")
            else: # freehand
                self.label.setText("Режим рисования включен: Свободное рисование. Удерживайте ЛКМ для рисования, отпустите для завершения контура.")
        else:
            self.label.setText("Режим рисования выключен. Используйте колесо мыши для масштабирования, ЛКМ для передвижения по изображению.")


    def save_contours_as_mask(self):
        """Сохраняет нарисованные контуры как бинарную маску."""
        if self.current_image_path == None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение, чтобы сохранить маску.")
            return

        # Получаем размеры исходного изображения
        try:
            with Image.open(self.current_image_path) as img:
                original_width, original_height = img.size
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось получить размеры исходного изображения: {e}")
            return

        contours = self.paint_widget.get_contours_in_image_coords()
        if contours == []:
            QMessageBox.information(self, "Информация", "Нет нарисованных контуров для сохранения.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Сохранить маску контуров", "", "TIFF Files (*.tif);;PNG Files (*.png);;All Files (*)")
        if save_path == None:
            return

        try:
            mask_image = self.image_changer.create_mask_from_contours(original_width, original_height, contours)
            mask_image.save(save_path)
            QMessageBox.information(self, "Успех", f"Маска контуров успешно сохранена в {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при сохранении маски: {e}")


def main():
    app = QApplication(sys.argv)
    window = OilApp()
    window.resize(1024, 768) # Увеличим начальный размер окна
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
