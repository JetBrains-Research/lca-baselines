  import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

class CustomGraphItem(pg.GraphItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setData(x=[], y=[], pen=pg.mkPen('w'))
        self.setLabel('left', 'Y Axis')
        self.setLabel('bottom', 'X Axis')
        self.setTitle('pyqtgraph example: CustomGraphItem')

    def setData(self, x, y, pen):
        self.x = x
        self.y = y
        self.pen = pen
        self.updateGraph()

    def updateGraph(self):
        self.setData(x=self.x, y=self.y, pen=self.pen)

    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.dragging = True
            self.dragStart = ev.pos()
            self.dragStartX = self.x[ev.pos().x()]
            self.dragStartY = self.y[ev.pos().x()]
        elif ev.button() == QtCore.Qt.RightButton:
            self.dragging = False

    def mouseMoveEvent(self, ev):
        if self.dragging:
            dx = ev.pos().x() - self.dragStart.x()
            dy = ev.pos().y() - self.dragStart.y()
            self.x[self.dragStartX] += dx
            self.y[self.dragStartY] += dy
            self.updateGraph()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            print('Node clicked at x={}, y={}'.format(ev.pos().x(), ev.pos().y()))

if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget()
    win.setWindowTitle('pyqtgraph example: CustomGraphItem')
    win.addItem(CustomGraphItem())
    win.show()
    app.exec_()