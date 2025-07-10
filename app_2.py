import sys
from PyQt6.QtWidgets import (QApplication, QLabel, QHBoxLayout, QMainWindow, QPushButton, QVBoxLayout, QWidget,
                             QTreeView, QFileDialog, QComboBox)
from PyQt6.QtCore import Qt, QPoint, QModelIndex, QRect
from PyQt6.QtGui import QFileSystemModel, QPainter, QPixmap, QMouseEvent, QWheelEvent


from backend_2 import *

class PaintWidget(QWidget):
    """
    Виджет для отображения изображений с возможностью
    масштабирования (зума) и панорамирования (перемещения).
    """
    def __init__(self):
        super().__init__()
        self.pixmap = QPixmap()
        self.scale_factor = 1.0  # Текущий масштаб изображения
        self.pan_start_position = QPoint() # Начальная точка для панорамирования
        self.pixmap_offset = QPoint() # Смещение изображения относительно центра
        self.is_panning = False # Флаг, указывающий, происходит ли сейчас панорамирование

        # Устанавливаем минимальный размер, чтобы виджет не был слишком маленьким
        self.setMinimumSize(600, 600)

        # Для рисования контуров
        self.drawing_enabled = False # Флаг: включен ли режим рисования
        self.is_drawing = False      # Флаг: происходит ли сейчас активное рисование
        self.current_contour = []    # Список QPoint для текущего рисуемого контура
        self.all_contours = []       # Список списков QPoint для всех завершенных контуров

    
    def enable_drawing(self):
        self.drawing_enabled = True
    
    def load_image(self, path):
        """Загружает новое изображение и сбрасывает трансформации."""
        if self.pixmap.load(path):
            # Сбрасываем масштаб и смещение при загрузке нового изображения
            self.scale_factor = 1.0
            self.pixmap_offset = QPoint()
            self.update()  # Запрашиваем перерисовку виджета
        else:
            print(f"Ошибка: не удалось загрузить изображение по пути {path}")


    def wheelEvent(self, event: QWheelEvent):
        """Обрабатывает событие прокрутки колеса мыши для масштабирования."""
        angle = event.angleDelta().y()
        # Определяем коэффициент масштабирования
        factor = 1.1 if angle > 0 else 1 / 1.1
        
        self.scale_factor *= factor
        self.update()  # Перерисовываем виджет с новым масштабом

    def mousePressEvent(self, event: QMouseEvent):
        """Обрабатывает нажатие кнопки мыши для начала панорамирования."""
        # Начинаем панорамирование по нажатию левой кнопки мыши
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.pan_start_position = event.pos()
        elif self.drawing_enabled == True and event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True
            self.current_contour = [event.pos()] # Начинаем новый контур с текущей позиции
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Обрабатывает движение мыши для перемещения изображения."""
        if self.is_panning:
            # Вычисляем смещение и обновляем позицию
            delta = event.pos() - self.pan_start_position
            self.pixmap_offset += delta
            self.pan_start_position = event.pos()
            self.update()  # Перерисовываем виджет в новом положении

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Обрабатывает отпускание кнопки мыши для завершения панорамирования."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False

    def paintEvent(self, event):
        """
        Перерисовывает виджет. Этот метод вызывается автоматически.
        Именно здесь происходит вся магия отрисовки с учетом масштаба и смещения.
        """
        super().paintEvent(event)
        
        if self.pixmap.isNull():
            return

        painter = QPainter(self)
        # Включаем сглаживание для более качественного отображения при масштабировании
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Вычисляем размеры и положение изображения с учетом масштаба
        target_width = int(self.pixmap.width() * self.scale_factor)
        target_height = int(self.pixmap.height() * self.scale_factor)

        # Центрируем изображение и применяем смещение от панорамирования
        widget_center = self.rect().center()
        pixmap_top_left = widget_center - QPoint(target_width // 2, target_height // 2) + self.pixmap_offset

        # Создаем прямоугольник, в котором будет нарисовано изображение
        target_rect = QRect(pixmap_top_left.x(), pixmap_top_left.y(), target_width, target_height)

        # Рисуем изображение
        painter.drawPixmap(target_rect, self.pixmap)




class OilApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('Идентификация нефтепродуктов на спутниковых снимках')
        self.image_changer = ImageChanger()
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
        self.tree.setFixedHeight(400)
        self.tree.clicked.connect(self.on_file_clicked)
        self.tree.hide() # Скрываем дерево, пока не выбрана папка
        # Скрываем лишние колонки (размер, тип, дата)
        for col in range(1, self.model.columnCount()):
            self.tree.hideColumn(col)

        self.instrument_label = QLabel('Инструменты для преобразования выбранных изображений')

        self.clear_button = QPushButton('Удалить всё из директории')
        self.clear_button.clicked.connect(self.clear_directory)

        self.find_edges_button = QPushButton('Найти границы на изображении')
        self.find_edges_button.clicked.connect(self.find_edges)

        self.geotiff_to_tiff_button = QPushButton('Преобразовать GeoTiff в Tiff')
        self.geotiff_to_tiff_button.clicked.connect(self.geotiff_to_tiff)

        self.crop_label = QLabel('Подготовка данных для обуения модели')

        # Кнопка для нарезания
        self.sum_channels_button = QPushButton('Сложение каналов изображения')
        self.sum_channels_button.clicked.connect(self.sum_channels)

        band_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']
        self.band1_box = QComboBox(self)
        self.band2_box = QComboBox(self)
        self.band1_box.addItems(band_list)
        self.band2_box.addItems(band_list)

        self.crop_button = QPushButton('Подготовить выборку')
        self.crop_button.clicked.connect(self.run_markup)

        layH2 = QHBoxLayout()
        layH2.addWidget(self.band1_box)
        layH2.addWidget(self.band2_box)

        # Собираем левую часть интерфейса
        layV1.addWidget(self.dir_button)
        layV1.addWidget(self.tree)
        layV1.addWidget(self.instrument_label)
        layV1.addWidget(self.clear_button)
        layV1.addWidget(self.find_edges_button)
        layV1.addWidget(self.geotiff_to_tiff_button)
        layV1.addWidget(self.crop_label)
        layV1.addLayout(layH2)
        layV1.addWidget(self.sum_channels_button)
        layV1.addWidget(self.crop_button)

        # Собираем основной горизонтальный слой
        layH.addLayout(layV1)
        layH.addWidget(self.paint_widget, 1) # Добавляем виджет рисования с растяжением

        # Собираем главный слой окна
        main_lay.addWidget(self.label)
        main_lay.addLayout(layH)
    
        # Виджет для рисования уже добавлен в layH, вторая вставка не нужна
        # main_lay.addWidget(self.paint_widget)

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
        input_path = QFileDialog.getExistingDirectory(self, 'Укажите путь к папке с файлами GeoTiff')
        output_path = QFileDialog.getExistingDirectory(self, 'Укажите путь, куда файл нужно сохранить')
        self.image_changer.geotiff_to_tiff(input_path, output_path)


    def find_edges(self):
        index = self.tree.currentIndex()
        if index.isValid() == False:
            print("Ничего не выбрано.")
            return
        input_path = self.model.filePath(index)
        output_path = QFileDialog.getExistingDirectory(self, 'Укажите путь, куда файл нужно сохранить')
        self.image_changer.find_edges(input_path, output_path, 'edges.tif')

    
    def sum_channels(self):
        input_path = self.dir_path
        band1, band2 = self.band1_box.currentText(), self.band2_box.currentText()
        output_path = QFileDialog.getExistingDirectory(self, 'Укажите путь, куда сохранить синтетическое изображение')
        self.image_changer.sum_channels(input_path, output_path, band1, band2)


    def clear_directory(self):
        directory = QFileDialog.getExistingDirectory(self, 'Выбор директории, которую нужно полностью очистить')
        print(os.listdir(directory))
        for dir in os.listdir(directory):
            try:
                shutil.rmtree(dir, ignore_errors=True)
            except Exception:
                pass


    def on_file_clicked(self, index: QModelIndex):
        file_path = self.model.filePath(index)
        # Проверяем, что это файл, а не папка
        if not self.model.isDir(index):
            self.paint_widget.load_image(file_path)


    def run_markup(self):
        snaps_path = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        masks_path = QFileDialog.getExistingDirectory(self, "Выберите папку с масками")
        save_dir = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if not snaps_path or not masks_path:
            return
        self.markup = ImageMarkup(snaps_path, masks_path)
        try:
            self.markup.work(save_dir)
            self.label.setText("Обработка завершена")
        except Exception as e:
            self.label.setText(f"Ошибка обработки: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = OilApp()
    window.resize(1024, 768) # Увеличим начальный размер окна
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
