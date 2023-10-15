from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QPainter
from PyQt6.QtCore import Qt, pyqtSignal

class View(QGraphicsView):
    zoom_signal = pyqtSignal(float)
    move_signal = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.dragging = False
        self.last_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.pos() - self.last_pos
            self.last_pos = event.pos()
            x = self.horizontalScrollBar().value() - delta.x()
            y = self.verticalScrollBar().value() - delta.y()
            self.move_signal.emit(x, y)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.last_pos = None

    def wheelEvent(self, event):
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        zoom_in = 1.15
        zoom_out = 1.0 / zoom_in
        delta = event.angleDelta().y() / 120
        if delta > 0:
            self.zoom_signal.emit(zoom_in)
        else:
            self.zoom_signal.emit(zoom_out)

