 ```python
import pyqtgraph as pg
from pyqtgraph.GraphItems import GraphItem
from pyqtgraph.Qt import QtCore, QtGui

class CustomGraphItem(GraphItem):
    def __init__(self, nodes, edges, **kwargs):
        super().__init__(**kwargs)
        self.nodes = nodes
        self.edges = edges
        self.data = {}
        self.setData(self.nodes, self.edges)

    def setData(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.data = {}
        for node in nodes:
            self.data[node] = []
        for edge in edges:
            self.data[edge[0]].append(edge[1])
        self.updateGraph()

    def updateGraph(self):
        self.clear()
        for node, edges in self.data.items():
            self.plot(x=node[0], y=node[1], symbol='o', symbolBrush=(node[2],), pen=None, name=f'Node {node}')
            for edge in edges:
                self.plot(x=[node[0], edge[0]], y=[node[1], edge[1]], symbol='', symbolBrush=None, pen='r', name=f'Edge {node}-{edge}')

    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            item = self.scene().itemAt(ev.pos())
            if item is not None and isinstance(item, pg.GraphicsObject):
                node = item.name().split()[1]
                self.nodes[int(node)] = (ev.pos().x(), ev.pos().y())
                self.setData(self.nodes, self.edges)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            item = self.scene().itemAt(ev.pos())
            if item is not None and isinstance(item, pg.GraphicsObject):
                node = item.name().split()[1]
                print(f'Node {node} clicked.')

if __name__ == '__main__':
    app = QtGui.QApplication([])
    view = pg.GraphicsView()
    layout = pg.GraphicsLayoutWidget()
    view.setCentralItem(layout)
    graph = CustomGraphItem(nodes=[(100, 100, 'r'), (300, 300, 'g'), (500, 100, 'b')], edges=[(0, 1), (1, 2)])
    layout.addItem(graph)
    view.show()
    app.exec_()
```