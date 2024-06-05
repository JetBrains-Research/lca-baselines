```python
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

class CustomGraphItem(pg.GraphItem):
    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        
    def setData(self, **kwds):
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.updateGraph()
        
    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        
    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        if ev.isStart():
            self.dragPoint = self.mapToParent(ev.buttonDownPos())
            self.dragOffset = self.dragPoint - self.data['pos']
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
            self.data['pos'] = self.mapToParent(ev.pos()) - self.dragOffset
            self.updateGraph()
            ev.accept()
            
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            clickedNode = self.scatter.pointsAt(ev.pos())
            if len(clickedNode) > 0:
                print("Node clicked:", clickedNode[0].data())
        
app = QtGui.QApplication([])
win = pg.GraphicsLayoutWidget()
win.setWindowTitle('pyqtgraph example: CustomGraphItem')
win.show()

g = CustomGraphItem()
win.addItem(g)

pos = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
adj = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
styles = [{'pen': {'color': 'r', 'width': 2}}, {'pen': {'color': 'g', 'width': 2}}, {'pen': {'color': 'b', 'width': 2}}, {'pen': {'color': 'y', 'width': 2}}]
texts = ['Node 1', 'Node 2', 'Node 3', 'Node 4']

g.setData(pos=pos, adj=adj, pen=styles, text=texts)

if __name__ == '__main__':
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
```