```python
import sys
from PyQt5 import QtGui
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

class CustomGraphItem(pg.GraphItem):
    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        super().__init__()

    def setData(self, **kwds):
        self.textItems = []
        for item in self.textItems:
            self.viewBox.removeItem(item)
        self.textItems = []
        super().setData(**kwds)
        for nodePos, text in zip(self.pos, kwds.get('texts', [])):
            item = pg.TextItem(text)
            self.textItems.append(item)
            item.setParentItem(self)
            item.setPos(nodePos[0], nodePos[1])

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the current mouse position and remember it(them).
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].index()
            self.dragOffset = self.pos[ind] - pos
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        ind = self.dragPoint.index()
        self.pos[ind] = ev.pos() + self.dragOffset
        self.setData(pos=self.pos)
        ev.accept()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            pos = ev.pos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) != 0:
                print("Node clicked:", pts[0].index())
                ev.accept()

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()

        self.graphWidget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.graphWidget)
        self.graphWidget.setWindowTitle('pyqtgraph example: CustomGraphItem')

        self.view = self.graphWidget.addViewBox()
        self.view.setAspectLocked()

        self.graph = CustomGraphItem()
        self.view.addItem(self.graph)

        pos = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5]], dtype=float)
        adj = np.array([
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [1, 2],
            [3, 0],
            [4, 0],
            [4, 1],
            [4, 2],
            [4, 3]
        ])
        texts = ['A', 'B', 'C', 'D', 'E']
        lines = np.array([
            (255, 0, 0, 255, 1),
            (0, 255, 0, 255, 2),
            (0, 0, 255, 255, 3),
            (255, 255, 0, 255, 2),
            (255, 0, 255, 255, 1),
            (0, 255, 255, 255, 3),
            (128, 128, 128, 255, 1),
            (128, 0, 0, 255, 2),
            (0, 128, 0, 255, 3),
            (0, 0, 128, 255, 2)
        ], dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte), ('width', float)])
        symbols = ['o', 's', 't', 'd', 'h']
        self.graph.setData(pos=pos, adj=adj, pen=lines, size=1, symbol=symbols, pxMode=False, texts=texts)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
```