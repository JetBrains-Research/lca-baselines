```python
import sys
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

class CustomGraphItem(pg.GraphItem):
    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        super().__init__()

    def setData(self, **kwds):
        self.textItems = []
        for item in self.textItems:
            self.scene().removeItem(item)
        self.textItems = []
        super().setData(**kwds)
        for node, pos in enumerate(self.data['pos']):
            text = pg.TextItem(self.data['texts'][node], anchor=(0.5, 0.5))
            text.setPos(pos[0], pos[1])
            self.textItems.append(text)
            self.viewBox.addItem(text)

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the current mouse position and remember it(them) for moving.
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = np.argwhere(self.data['pos'] == self.dragPoint.pos())[0][0]
            self.dragOffset = self.data['pos'][ind] - pos
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
            self.data['pos'][np.argwhere(self.data['pos'] == self.dragPoint.pos())[0][0]] = ev.pos() + self.dragOffset
            self.setData(pos=self.data['pos'], adj=self.data['adj'], pen=self.data['pen'], symbol=self.data['symbol'], size=self.data['size'], pxMode=self.data['pxMode'], texts=self.data['texts'])
            ev.accept()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            pos = ev.pos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) != 0:
                print("Node clicked:", pts[0].data())

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()

        self.graphWidget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.graphWidget)
        self.setWindowTitle('pyqtgraph example: CustomGraphItem')

        self.view = self.graphWidget.addViewBox()
        self.view.setAspectLocked()

        self.graph = CustomGraphItem()
        self.view.addItem(self.graph)

        pos = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5]], dtype=float)
        adj = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [1, 4], [2, 4], [3, 4]])
        texts = ['Node 1', 'Node 2', 'Node 3', 'Node 4', 'Node 5']
        self.graph.setData(pos=pos, adj=adj, pen=pg.mkPen(width=2, color='r'), symbol='o', size=10, pxMode=False, texts=texts)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
```