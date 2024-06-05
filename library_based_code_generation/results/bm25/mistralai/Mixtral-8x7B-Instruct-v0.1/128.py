from nodegraphqt import NodeGraphQt

class NodeGraph:
def __init__(self):
self.graph = NodeGraphQt()

# Zoom functions
def zoom_in(self):
self.graph.set_zoom(self.graph.get_zoom() * 1.2)

def zoom_out(self):
self.graph.set_zoom(self.graph.get_zoom() / 1.2)

def reset_zoom(self):
self.graph.reset_zoom()

def fit_selected_nodes(self):
self.graph.fit_to_selection()

# Layout functions
def set_layout_horizontal(self):
self.graph.set_layout_direction("LEFT_RIGHT")

def set_layout_vertical(self):
self.graph.set_layout_direction("TOP_BOTTOM")

# File functions
def open_session(self, filename):
self.graph.open_session(filename)

def import_session(self, filename):
self.graph.import_session(filename)

def save_session(self, filename):
self.graph.save_session(filename)

def clear_session(self):
self.graph.clear_session()

# Node functions
def clear(self):
self.graph.clear()

def clear_connections(self, node):
self.graph.clear_connections(node)

def clear_undo_history(self):
self.graph.clear_undo_stack()

def copy(self):
self.graph.copy()

def cut(self):
self.graph.cut()

def paste(self, position):
self.graph.paste(position)

def delete(self):
self.graph.delete()

def extract(self, node):
self.graph.extract(node)

def select_all(self):
self.graph.select_all()

def clear_selection(self):
self.graph.clear_selection()

def invert_selection(self):
self.graph.invert_selection()

def disable(self, node):
self.graph.disable(node)

def duplicate(self, node):
self.graph.duplicate(node)

def expand_group(self, node):
self.graph.expand_group(node)

def fit_zoom_to_selected_nodes(self):
self.graph.zoom_to_nodes()

# Display functions
def show_undo_list(self):
self.graph.show_undo_list()

# Pipe style functions
def set_pipe_style_curved(self):
self.graph.set_pipe_layout("curved")

def set_pipe_style_straight(self):
self.graph.set_pipe_layout("straight")

def set_pipe_style_angled(self):
self.graph.set_pipe_layout("angled")

# Grid style functions
def set_grid_style_none(self):
self.graph.set_grid_style("none")

def set_grid_style_dots(self):
self.graph.set_grid_style("dots")

def set_grid_style_lines(self):
self.graph.set_grid_style("lines")

# Auto layout functions
def auto_layout_downstream(self):
self.graph.auto_layout_nodes("downstream")

def auto_layout_upstream(self):
self.graph.auto_layout_nodes("upstream")

# Search functions
def toggle_node_search(self):
self.graph.toggle_node_search()