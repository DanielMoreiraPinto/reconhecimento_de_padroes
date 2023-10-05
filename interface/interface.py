from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView, QPushButton, QFileDialog, \
    QMainWindow, QApplication, QStatusBar, QWidget, QLabel, QSizePolicy
from PyQt6.QtGui import QPixmap, QFont, QIcon
from PyQt6.QtCore import QDir, QSize, QRect, QCoreApplication, QMetaObject
import os
from setup_path import folder_relative_path
from view import View


class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):

        QDir.addSearchPath('icons', os.path.join(folder_relative_path, 'icons'))

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 700)
        MainWindow.setFixedSize(1100, 700)

        self.centralwidget = QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.scene_img_original = QGraphicsScene()
        self.view_img_original = View(parent=self.centralwidget)
        self.view_img_original.setGeometry(QRect(20, 50, 440, 540))  
        self.view_img_original.setScene(self.scene_img_original)
        self.view_img_original.setObjectName("view_img_original")

        self.scene_img_resultante = QGraphicsScene()
        self.view_img_resultante = View(parent=self.centralwidget)
        self.view_img_resultante.setGeometry(QRect(470, 50, 440, 540))
        self.view_img_resultante.setScene(self.scene_img_resultante)
        self.view_img_resultante.setObjectName("view_img_resultante")

        ## para mover e dar zoom nas duas views ao mesmo tempo
        self.view_img_original.zoom_signal.connect(self.apply_zoom)
        self.view_img_resultante.zoom_signal.connect(self.apply_zoom)
        self.view_img_original.move_signal.connect(self.apply_movement)
        self.view_img_resultante.move_signal.connect(self.apply_movement)

        self.font_sub = QFont()
        self.font_sub.setFamily("Montserrat")
        self.font_sub.setPointSize(12)

        self.font_title = QFont()
        self.font_title.setFamily("Montserrat")
        self.font_title.setPointSize(18)

        self.label = QLabel(parent=self.centralwidget)
        self.label.setGeometry(QRect(160, 610, 260, 50))
        self.label.setObjectName("label")
        self.label.setFont(self.font_sub)

        self.label_2 = QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QRect(620, 610, 260, 50))
        self.label_2.setObjectName("label_2")
        self.label_2.setFont(self.font_sub)

        self.label_3 = QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QRect(930, 170, 190, 31))
        self.label_3.setObjectName("label_3")
        self.label_3.setFont(self.font_title)

        self.label_4 = QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QRect(160, 10, 260, 30))
        self.label_4.setObjectName("label_4")
        self.label_4.setFont(self.font_title)

        self.label_5 = QLabel(parent=self.centralwidget)
        self.label_5.setGeometry(QRect(620, 10, 260, 30))
        self.label_5.setObjectName("label_5")
        self.label_5.setFont(self.font_title)

        icon_reduzir_ruido = QIcon()
        icon_reduzir_ruido.addPixmap(QPixmap("icons:denoising.png"), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton = QPushButton(parent=self.centralwidget)
        self.pushButton.setGeometry(QRect(930, 210, 60, 60))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setIcon(icon_reduzir_ruido)
        self.pushButton.setIconSize(QSize(self.pushButton.sizeHint().width(), self.pushButton.sizeHint().height()))
        self.pushButton.setToolTip("Reduzir Ruído")

        icon_salvar_imagem = QIcon()
        icon_salvar_imagem.addPixmap(QPixmap("icons:save-result.png"), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_2 = QPushButton(parent=self.centralwidget)
        self.pushButton_2.setGeometry(QRect(1000, 50, 60, 60))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setIcon(icon_salvar_imagem)
        self.pushButton_2.setIconSize(QSize(self.pushButton_2.sizeHint().width(), self.pushButton_2.sizeHint().height()))
        self.pushButton_2.setToolTip("Salvar Imagem")
        self.pushButton_2.clicked.connect(self.on_botao_resultante_clicked)

        icon_carregar_imagem = QIcon()
        icon_carregar_imagem.addPixmap(QPixmap("icons:load_image.png"), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_3 = QPushButton(parent=self.centralwidget)
        self.pushButton_3.setGeometry(QRect(930, 50, 60, 60))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setIcon(icon_carregar_imagem)
        self.pushButton_3.setIconSize(QSize(self.pushButton_3.sizeHint().width(), self.pushButton_3.sizeHint().height()))
        self.pushButton_3.setToolTip("Adicionar Imagem")
        self.pushButton_3.clicked.connect(self.on_botao_original_clicked)

        icon_reduzir_borrao = QIcon()
        icon_reduzir_borrao.addPixmap(QPixmap("icons:sharpening.png"), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_4 = QPushButton(parent=self.centralwidget)
        self.pushButton_4.setGeometry(QRect(1000, 210, 60, 60))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setIcon(icon_reduzir_borrao)
        self.pushButton_4.setIconSize(QSize(self.pushButton_4.sizeHint().width(), self.pushButton_4.sizeHint().height()))
        self.pushButton_4.setToolTip("Aumentar Nitidez")

        icon_aumentar_resolucao = QIcon()
        icon_aumentar_resolucao.addPixmap(QPixmap("icons:super-resolution.png"), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_5 = QPushButton(parent=self.centralwidget)
        self.pushButton_5.setGeometry(QRect(930, 280, 60, 60))
        self.pushButton_5.setIcon(icon_aumentar_resolucao)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_5.sizePolicy().hasHeightForWidth())
        self.pushButton_5.setSizePolicy(sizePolicy)
        self.pushButton_5.setIconSize(QSize(self.pushButton_5.sizeHint().width(), self.pushButton_5.sizeHint().height()))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.setToolTip("Aumentar Resolução")

        icon_transferir_estilo = QIcon()
        icon_transferir_estilo.addPixmap(QPixmap("icons:style-transfer.png"), QIcon.Mode.Normal, QIcon.State.Off)

        self.pushButton_6 = QPushButton(parent=self.centralwidget)
        self.pushButton_6.setGeometry(QRect(1000, 280, 60, 60))
        self.pushButton_6.setIcon(icon_transferir_estilo)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_6.sizePolicy().hasHeightForWidth())
        self.pushButton_6.setIconSize(QSize(self.pushButton_6.sizeHint().width(), self.pushButton_6.sizeHint().height()))
        self.pushButton_6.setSizePolicy(sizePolicy)
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.setToolTip("Transferir Estilo")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)
    
    def load_image(self, scene):
        file_name, _ =  QFileDialog.getOpenFileName(self, 'Selecione uma imagem...', '.', 'Image files (*.jpg *.gif *.png *.jpeg)')
        if file_name:
            pixmap = QPixmap(file_name)
            if not pixmap.isNull(): 
                print('o tipo dessa desgraça', type(scene))
                scene.clear()
                scene.addPixmap(pixmap)
    
    def on_botao_resultante_clicked(self):
        self.load_image(self.scene_img_resultante)
    
    def on_botao_original_clicked(self):
        self.load_image(self.scene_img_original)

    def apply_zoom(self, zoom_factor):
        self.view_img_original.scale(zoom_factor, zoom_factor)
        self.view_img_resultante.scale(zoom_factor, zoom_factor)
    
    def apply_movement(self, move_factor_x, move_factor_y):
        self.view_img_original.horizontalScrollBar().setValue(move_factor_x)
        self.view_img_original.verticalScrollBar().setValue(move_factor_y)
        self.view_img_resultante.horizontalScrollBar().setValue(move_factor_x)
        self.view_img_resultante.verticalScrollBar().setValue(move_factor_y)

    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Free AI Tools"))
        # self.pushButton.setText(_translate("MainWindow", "Reduzir Ruído"))
        self.label.setText(_translate("MainWindow", "Nível de Comprometimento: "))
        self.label_2.setText(_translate("MainWindow", "Nível de Comprometimento: "))
        # self.pushButton_2.setText(_translate("MainWindow", "Salvar Resultado"))
        # self.pushButton_3.setText(_translate("MainWindow", "Carregar Imagem"))
        self.label_3.setText(_translate("MainWindow", "Ferramentas: "))
        # self.pushButton_4.setText(_translate("MainWindow", "Reduzir Borrão"))
        # self.pushButton_5.setText(_translate("MainWindow", "Aumentar Resolução"))
        # self.pushButton_6.setText(_translate("MainWindow", "Transferir Estilo"))
        self.label_4.setText(_translate("MainWindow", "Imagem Original"))
        self.label_5.setText(_translate("MainWindow", "Imagem Resultante"))


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet("""
    QWidget {
        background-color: #333333;
        color: #ffffff;
    }
    QPushButton {
        background-color: #6F42C1;
        color: #FFFFFF;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #5A32A3;
    }   
    QLabel {
        
    }
    QGraphicsView {
        background-color: #696969;
    }
    """)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
