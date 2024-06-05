 import pyqtgraph as pg
from pyqtgraph.GraphItems import GraphItem, NodeGraphicsItem

class CustomGraphItem(GraphItem):
def __init__(self, nodes_positions, nodes_labels, nodes_connections):
super().__init__()
self.nodes_positions = nodes_positions
self.nodes_labels = nodes_labels
self.nodes_connections = nodes_connections
self.nodes = []
self.create_nodes()
self.update_graph()

def create_nodes(self):
for pos, label in zip(self.nodes_positions, self.nodes_labels):
node = NodeGraphicsItem(pos=pos, rotate=0, border=2, pen=pg.mkPen('b'))
node.setLabel(label)
node.setFlag(NodeGraphicsItem.ItemIsMovable, True)
node.setFlag(NodeGraphicsItem.ItemIsSelectable, True)
self.nodes.append(node)

def update_graph(self):
self.clear()
for node in self.nodes:
self.addItem(node)
for connection in self.nodes_connections:
line = pg.PlotDataItem([node.pos()[0], connection[1][0]],
[node.pos()[1], connection[1][1]],
pen=pg.mkPen('r'))
self.addItem(line)

def handle_mouse_drag_event(self, event):
if event.button() == Qt.LeftButton:
for node in self.nodes:
if node.sceneBoundingRect().contains(event.scenePos()):
node.setPos(event.scenePos())
self.update_graph()

def handle_click_event(self, event):
print("Node clicked!")

def set_data(self, nodes_positions, nodes_labels, nodes_connections):
self.nodes_positions = nodes_positions
self.nodes_labels = nodes_labels
self.nodes_connections = nodes_connections
self.clear()
self.create_nodes()
self.update_graph()

app = pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.setWindowTitle('pyqtgraph example: CustomGraphItem')
graph = CustomGraphItem(nodes_positions=[(100, 100), (300, 300)],
nodes_labels=['A', 'B'],
nodes_connections=[[(100, 100), (300, 300)]])
win.addItem(graph)
win.show()
app.exec_()