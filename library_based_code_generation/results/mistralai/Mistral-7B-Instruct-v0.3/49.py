 Here is a basic implementation of the requested custom graph using pyqtgraph:

```python
import sys
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.graphicsItems import GraphItem, LineItem, TextItem

class CustomGraphItem(GraphItem):
    def __init__(self):
        super().__init__()
        self.nodes = {}
        self.lines = {}
        self.selected_node = None

    def set_data(self, nodes, edges):
        self.clear()
        for node, position in nodes.items():
            self.add_node(node, position)
        for edge, (node1, node2) in edges.items():
            self.add_edge(node1, node2)

    def add_node(self, node, position):
        text_item = TextItem(node)
        text_item.setPos(*position)
        self.nodes[node] = text_item
        self.addItem(text_item)
        node_item = self.addRect(position[0] - 10, position[1] - 10, 20, 20, pen='r', brush='w')
        node_item.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        node_item.setZValue(1)

    def add_edge(self, node1, node2):
        if node1 not in self.nodes or node2 not in self.nodes:
            return
        line = LineItem((self.nodes[node1].pos(), self.nodes[node2].pos()), pen='b')
        self.lines[(node1, node2)] = line
        self.addItem(line)

    def update_graph(self):
        for node, text_item in self.nodes.values():
            text_item.setPos(self.mapToScene(text_item.pos()))
        for line in self.lines.values():
            line.setData((self.mapFromScene(line.pos1()), self.mapFromScene(line.pos2())))

    def mouseDragEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            for node, item in self.nodes.items():
                if item.boundingRect().contains(event.pos()):
                    self.selected_node = node
                    break
            if self.selected_node:
                self.update_graph()

    def sceneMouseClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            for node, item in self.nodes.items():
                if item.boundingRect().contains(event.pos()):
                    print(f'Node {node} clicked')
                    self.selected_node = node
                    break
            self.update_graph()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    gl = GraphicsLayoutWidget()
    graph = CustomGraphItem()

    nodes = {'A': (100, 100), 'B': (200, 200), 'C': (300, 300)}
    edges = {'AB': ('A', 'B'), 'BC': ('B', 'C')}
    graph.set_data(nodes, edges)

    gl.addItem(graph)
    win.setCentralWidget(gl)
    win.show()
    sys.exit(app.exec_())
```

This code creates a custom graph using the pyqtgraph library, with nodes and edges that can be dragged and clicked. The graph is displayed in a GraphicsLayoutWidget, and the script prints a message to the console when a node is clicked.