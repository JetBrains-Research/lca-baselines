 Here is a basic structure for the functions you requested using the `nodegraphqt` library. Note that this is a simplified version and might need adjustments based on the actual library and your specific requirements.

```python
from nodegraphqt import GraphView, Node, Edge, GraphScene

class NodeGraph:
    def __init__(self, parent=None):
        self.graph = GraphView()
        self.scene = GraphScene(self.graph, self)

    def zoom_in(self):
        self.graph.setZoom(self.graph.zoomFactor() * 1.25)

    def zoom_out(self):
        self.graph.setZoom(self.graph.zoomFactor() / 1.25)

    def reset_zoom(self):
        self.graph.setZoom(1.0)

    def set_layout_direction(self, direction):
        self.scene.setLayoutDirection(direction)

    def open_session(self, filename):
        self.graph.load(filename)

    def import_session(self, data):
        self.graph.importGraph(data)

    def save_session(self, filename):
        self.graph.save(filename)

    def clear_session(self):
        self.graph.clear()

    def clear_undo_history(self):
        self.graph.undoStack().clear()

    def copy(self):
        self.graph.copy()

    def cut(self):
        self.graph.cut()

    def paste(self):
        self.graph.paste()

    def delete(self, node):
        node.scene().removeItem(node)

    def extract(self, node):
        return self.graph.extract(node)

    def clear_connections_of(self, node):
        for edge in node.outgoingEdges():
            self.delete(edge)
        for edge in node.incomingEdges():
            self.delete(edge)

    def select_all(self):
        self.graph.selectAll()

    def clear_selection(self):
        self.graph.clearSelection()

    def invert_selection(self):
        self.graph.invertSelection()

    def disable(self, node):
        node.setEnabled(False)

    def duplicate(self, node):
        new_node = node.copy()
        self.graph.addItem(new_node)

    def expand_group(self, group):
        group.setExpanded(True)

    def fit_zoom_level_to_selected_nodes(self):
        self.graph.fitInView(self.graph.selectedNodes(), QtCore.Qt.KeepAspectRatio)

    def show_undo_list(self):
        self.graph.undoStack().show()

    def set_pipe_style(self, style):
        self.graph.setPipeStyle(style)

    def set_background_grid(self, grid_type):
        self.scene.setGridVisible(True)
        if grid_type == 'none':
            self.scene.setGridDashPattern([])
        elif grid_type == 'dots':
            self.scene.setGridDashPattern([2, 2])
        elif grid_type == 'lines':
            self.scene.setGridDashPattern([10, 10])

    def auto_layout_nodes_downstream(self):
        self.scene.autoLayoutDownstream()

    def auto_layout_nodes_upstream(self):
        self.scene.autoLayoutUpstream()

    def toggle_node_search(self):
        self.graph.toggleNodeSearch()
```

This code creates a `NodeGraph` class that contains methods for most of the functionalities you requested. However, it's important to note that the `nodegraphqt` library might have specific requirements or methods for certain functionalities, so you should refer to the library's documentation for more accurate implementations.