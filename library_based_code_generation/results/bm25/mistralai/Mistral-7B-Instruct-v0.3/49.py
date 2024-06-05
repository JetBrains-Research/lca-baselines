 Here is a basic implementation of the custom graph using pyqtgraph:

```python
import sys
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.graphicsItems import GraphItem, GraphIcon, GLBarGraphItem

class CustomGraphItem(GraphItem):
    def __init__(self):
        super().__init__()
        self.nodes = {}
        self.connections = {}
        self.title = 'pyqtgraph example: CustomGraphItem'

    def set_data(self, nodes, connections):
        self.clear()
        self.nodes = nodes
        self.connections = connections
        for node, position in nodes.items():
            icon = GraphIcon(GraphIcon.Ellipse, brush=QtGui.QColor('red'), pen=QtGui.QPen(QtCore.Qt.black, 2))
            icon.setPos(*position)
            icon.setLabel(node, anchor=QtCore.Qt.TextAnchor.Center)
            self.addItem(icon)
            icon.setDraggable(True)
            self.nodes[node] = icon

        for connection in connections:
            start_node, end_node, style = connection
            start_icon = self.nodes[start_node]
            end_icon = self.nodes[end_node]
            bar = GLBarGraphItem(start=start_icon.pos(), end=end_icon.pos(), orientation='horizontal', pen=style)
            self.addItem(bar)
            self.connections[(start_node, end_node)] = bar

    def update_graph(self):
        for icon in self.items():
            if isinstance(icon, GraphIcon):
                icon.setPos(*icon.data['pos'])

    def mouseDragEvent(self, event):
        for icon in self.items():
            if isinstance(icon, GraphIcon) and icon.isDraggable():
                icon.setPos(event.pos())
                self.update_graph()

    def sceneMouseClicked(self, event):
        for icon in self.items():
            if isinstance(icon, GraphIcon):
                if icon.boundingRect().contains(event.pos()):
                    print(f'Node {icon.label()} clicked')

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    win = QtWidgets.QGraphicsView()
    layout = GraphicsLayoutWidget()
    layout.addItem(CustomGraphItem(), colspan=2)
    win.setCentralItem(layout)
    win.show()
    sys.exit(app.exec_())
```

This code creates a custom graph item that can be set with data, updates in real-time as nodes are dragged, and responds to clicks on the graph. The graph is displayed in a GraphicsLayoutWidget and the program prints a message to the console when a node is clicked.