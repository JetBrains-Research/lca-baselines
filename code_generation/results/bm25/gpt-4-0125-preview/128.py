```python
from nodegraphqt import BaseNodeGraph

def zoom_in(graph: BaseNodeGraph, factor=1.1):
    current_zoom = graph.get_zoom()
    graph.set_zoom(current_zoom * factor)

def zoom_out(graph: BaseNodeGraph, factor=0.9):
    current_zoom = graph.get_zoom()
    graph.set_zoom(current_zoom * factor)

def reset_zoom(graph: BaseNodeGraph):
    graph.reset_zoom()

def set_layout_direction(graph: BaseNodeGraph, direction='horizontal'):
    graph.set_layout_direction(direction)

def open_session(graph: BaseNodeGraph, filepath):
    graph.load_session(filepath)

def import_session(graph: BaseNodeGraph, filepath):
    graph.import_session(filepath)

def save_session(graph: BaseNodeGraph, filepath):
    graph.save_session(filepath)

def clear_session(graph: BaseNodeGraph):
    graph.clear_session()

def clear_undo_history(graph: BaseNodeGraph):
    graph.clear_undo_stack()

def copy(graph: BaseNodeGraph):
    graph.copy_nodes()

def cut(graph: BaseNodeGraph):
    graph.cut_nodes()

def paste(graph: BaseNodeGraph):
    graph.paste_nodes()

def delete(graph: BaseNodeGraph):
    graph.delete_nodes()

def extract(graph: BaseNodeGraph):
    graph.extract_nodes()

def clear_connections(graph: BaseNodeGraph):
    graph.clear_connections()

def select_all(graph: BaseNodeGraph):
    graph.select_all()

def clear_selection(graph: BaseNodeGraph):
    graph.clear_selection()

def invert_selection(graph: BaseNodeGraph):
    graph.invert_selection()

def disable(graph: BaseNodeGraph):
    graph.disable_nodes()

def duplicate(graph: BaseNodeGraph):
    graph.duplicate_nodes()

def expand_group(graph: BaseNodeGraph):
    graph.expand_group()

def fit_zoom_to_selected_nodes(graph: BaseNodeGraph):
    graph.fit_to_selection()

def show_undo_list(graph: BaseNodeGraph):
    print(graph.undo_stack())

def set_pipe_style(graph: BaseNodeGraph, style):
    graph.set_pipe_layout(style)

def set_background_grid(graph: BaseNodeGraph, style):
    graph.viewer().set_grid_mode(style)

def auto_layout_nodes(graph: BaseNodeGraph, direction):
    graph.auto_layout_nodes(direction)

def toggle_node_search(graph: BaseNodeGraph):
    graph.show_search()
```