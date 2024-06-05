 Here is a basic structure for the functions you requested using the `nodegraphqt` library. Note that this is a simplified version and might need adjustments based on the actual library structure and your specific project requirements.

```python
from nodegraphqt import NodeGraph, Node, Edge, Viewer

class NodeGraphManager:
    def __init__(self):
        self.ng = NodeGraph()
        self.viewer = Viewer(self.ng)

    def zoom_in(self):
        self.viewer._set_viewer_zoom(self.viewer.get_zoom() * 1.1)

    def zoom_out(self):
        self.viewer._set_viewer_zoom(self.viewer.get_zoom() / 1.1)

    def reset_zoom(self):
        self.viewer.reset_zoom()

    def set_layout_direction(self, direction):
        self.ng.set_layout_direction(direction)

    def open_session(self, filename):
        self.ng.open(filename)

    def import_session(self, data):
        self.ng.import_from_dict(data)

    def save_session(self, filename):
        self.ng.save(filename)

    def clear_session(self):
        self.ng.clear()

    def clear_undo_history(self):
        self.ng.clear_undo_stack()

    def copy(self):
        self.ng.copy()

    def cut(self):
        self.ng.cut()

    def paste(self):
        self.ng.paste()

    def delete(self):
        self.ng.delete()

    def extract(self, nodes):
        extracted_nodes = [node for node in nodes if node.is_selected()]
        self.ng.clear_selection()
        for node in extracted_nodes:
            node.set_selected(True)
        return extracted_nodes

    def clear_connections_of(self, node):
        node.clear_connections()

    def select_all(self):
        self.ng.select_all()

    def clear_selection(self):
        self.ng.clear_selection()

    def invert_selection(self):
        self.ng.invert_selection()

    def disable(self, node):
        node.set_disabled(True)

    def duplicate(self, node):
        new_node = node.duplicate()
        self.ng.add_node(new_node)

    def expand_group(self, group):
        group.expand()

    def fit_to_selected_nodes(self):
        self.ng.zoom_to_nodes(self.ng.get_selected_nodes())

    def show_undo_list(self):
        self.ng.show_undo_list()

    def set_pipe_style(self, style):
        self.ng.set_pipe_layout(style)

    def set_background_grid(self, grid_type):
        self.viewer.set_background_grid(grid_type)

    def auto_layout_nodes(self, direction):
        self.ng.auto_layout_nodes(direction)

    def toggle_node_search(self):
        self.viewer._on_focus_to_node(None)

    def clear_registered_nodes(self):
        self.ng._clear_actions()

    def clear_bin(self):
        self.ng.clear_bin()
```

This code creates a `NodeGraphManager` class that provides methods for managing a `NodeGraph` instance and its associated functions. The class uses the provided APIs from the `nodegraphqt` library to perform the desired operations.