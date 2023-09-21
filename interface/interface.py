import sys
sys.path.append('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes')

import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QFrame  # Importe QFrame
from PyQt5.QtGui import QImage, QPixmap

from denoiser.denoiser import denoise


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Processamento de Imagem")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.original_image = None
        self.noise_image = None

        self.current_frame = None

        self.page = 1  # 1 para carregar imagem, 2 para processar imagem

        self.load_button = QPushButton("Carregar Imagem", self)
        self.load_button.clicked.connect(self.loadImage)

        self.detect_button = QPushButton("Detectar Ruído", self)
        self.detect_button.clicked.connect(self.detectNoise)
        self.remove_button = QPushButton("Remover Ruído", self)
        self.remove_button.clicked.connect(self.removeNoise)
        self.download_button = QPushButton("Download da Imagem sem Ruído", self)
        self.download_button.clicked.connect(self.saveImage)
        self.download_button.setEnabled(False)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.label_original = QLabel(self)
        self.label_processed = QLabel(self)

        self.layout.addWidget(self.load_button)

    def clear_layout(self):
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget is not None:
                self.layout.removeWidget(widget)
                widget.deleteLater()

    # def show_page(self):
    #     self.clear_layout()

    #     self.layout.addWidget(self.detect_button)
    #     self.layout.addWidget(self.remove_button)
    #     self.layout.addWidget(self.download_button)

    #     # Moldura para imagem carregada
    #     frame_original = QLabel(self)
    #     frame_original.setFixedSize(300, 300)
    #     frame_original.setFrameShape(QFrame.Box)  # Adicione moldura
    #     self.layout.addWidget(frame_original)

    #     # Moldura para imagem sem ruído
    #     frame_processed = QLabel(self)
    #     frame_processed.setFixedSize(300, 300)
    #     frame_processed.setFrameShape(QFrame.Box)  # Adicione moldura
    #     self.layout.addWidget(frame_processed)

    #     # Exibe as imagens dentro das molduras
    #     self.show_image(self.original_image, frame_original)
    #     self.show_image(self.noise_image, frame_processed)

    def show_page(self):
        # Create or update the main buttons
        self.layout.addWidget(self.detect_button)
        self.layout.addWidget(self.remove_button)
        self.layout.addWidget(self.download_button)

        # Create or update the frame for the original image
        frame_original = self.layout.findChild(QLabel, "frame_original")
        if frame_original is None:
            frame_original = QLabel(self)
            frame_original.setFixedSize(300, 300)
            frame_original.setFrameShape(QFrame.Box)
            frame_original.setObjectName("frame_original")
            self.layout.addWidget(frame_original)
        self.show_image(self.original_image, frame_original)

        # Create or update the frame for the processed image
        frame_processed = self.layout.findChild(QLabel, "frame_processed")
        if frame_processed is None:
            frame_processed = QLabel(self)
            frame_processed.setFixedSize(300, 300)
            frame_processed.setFrameShape(QFrame.Box)
            frame_processed.setObjectName("frame_processed")
            self.layout.addWidget(frame_processed)
        self.show_image(self.noise_image, frame_processed)

    def loadImage(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(filter="Image files (*.png *.jpg *.bmp *.jpeg *.gif *.tiff)")
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.page = 2
            self.show_page()

    def show_image(self, cv_image, label_widget):
        if cv_image is not None:
            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            label_widget.setPixmap(pixmap)
            label_widget.setScaledContents(True)

    def detectNoise(self):
        # Implemente a detecção de ruído aqui usando o modelo de detecção de ruído
        # Atualize a variável self.noise_image com a imagem detectada
        pass

    def removeNoise(self):
        self.noise_image = denoise(self.original_image)
        self.show_page()
        self.download_button.setEnabled(True)

    def saveImage(self):
        if self.noise_image is not None:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_path, _ = QFileDialog.getSaveFileName(self, "Salvar Imagem", "", "PNG Files (*.png)", options=options)
            
            if file_path:
                if not file_path.endswith(".png"):
                    file_path = file_path + ".png"
                cv2.imwrite(file_path, self.noise_image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())