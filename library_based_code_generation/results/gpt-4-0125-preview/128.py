```python
from nodegraphqt import BaseNodeGraph

def zoom_in(graph: BaseNodeGraph):
    graph.zoom_in()

def zoom_out(graph: BaseNodeGraph):
    graph.zoom_out()

def reset_zoom(graph: BaseNodeGraph):
    graph.reset_zoom()

def set_layout_direction(graph: BaseNodeGraph, direction: str):
    if direction.lower() == 'horizontal':
        graph.layout_direction = graph.HORIZONTAL
    elif direction.lower() == 'vertical':
        graph.layout_direction = graph.VERTICAL

def open_session(graph: BaseNodeGraph, filepath: str):
    graph.load_session(filepath)

def import_session(graph: BaseNodeGraph, filepath: str):
    graph.import_session(filepath)

def save_session(graph: BaseNodeGraph, filepath: str):
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
    for node in graph.selected_nodes():
        node.set_disabled(not node.disabled())

def duplicate(graph: BaseNodeGraph):
    graph.duplicate_nodes()

def expand_group(graph: BaseNodeGraph):
    for node in graph.selected_nodes():
        if node.type_() == 'GroupNode':
            node.expand()

def fit_zoom_to_nodes(graph: BaseNodeGraph):
    graph.fit_to_selection()

def show_undo_list(graph: BaseNodeGraph):
    print(graph.undo_stack())

def set_pipe_style(graph: BaseNodeGraph, style: str):
    if style.lower() == 'curved':
        graph.set_pipe_style(graph.PIPE_STYLE_CURVED)
    elif style.lower() == 'straight':
        graph.set_pipe_style(graph.PIPE_STYLE_STRAIGHT)
    elif style.lower() == 'angled':
        graph.set_pipe_style(graph.PIPE_STYLE_ANGLE)

def set_background_grid(graph: BaseNodeGraph, style: str):
    if style.lower() == 'none':
        graph.set_grid_mode(graph.GRID_MODE_NONE)
    elif style.lower() == 'dots':
        graph.set_grid_mode(graph.GRID_MODE_DOT)
    elif style.lower() == 'lines':
        graph.set_grid_mode(graph.GRID_MODE_LINE)

def auto_layout(graph: BaseNodeGraph, direction: str):
    if direction.lower() == 'downstream':
        graph.auto_layout_nodes(selected_only=False, down_stream=True)
    elif direction.lower() == 'upstream':
        graph.auto_layout_nodes(selected_only=False, down_stream=False)

def toggle_node_search(graph: BaseNodeGraph):
    graph.show_node_searcher(not graph.is_node_searcher_visible())
```