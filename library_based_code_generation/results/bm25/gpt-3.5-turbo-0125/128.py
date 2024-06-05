```python
def zoom_in():
    nodegraphqt.zoom_to_nodes()

def zoom_out():
    nodegraphqt.set_zoom(nodegraphqt.get_zoom() - 0.1)

def reset_zoom():
    nodegraphqt.reset_zoom()

def set_layout_direction(direction):
    nodegraphqt.set_layout_direction(direction)

def open_session():
    nodegraphqt.clear_session()

def import_session():
    nodegraphqt.clear_registered_nodes()

def save_session():
    nodegraphqt.to_dict()

def clear_session():
    nodegraphqt.clear()

def clear_undo_history():
    nodegraphqt.clear_undo_stack()

def copy_nodes():
    nodegraphqt.copy()

def cut_nodes():
    nodegraphqt.cut()

def paste_nodes():
    nodegraphqt.paste()

def delete_nodes():
    nodegraphqt.clear_selection()

def extract_nodes():
    nodegraphqt.clear_selection()

def clear_connections():
    nodegraphqt.clear_connections()

def select_all_nodes():
    nodegraphqt._on_focus_to_node()

def clear_selection():
    nodegraphqt.clear_selection()

def invert_selection():
    nodegraphqt.clear_selection()

def disable_nodes():
    nodegraphqt.clear_selection()

def duplicate_nodes():
    nodegraphqt.clear_selection()

def expand_group():
    nodegraphqt.clear_selection()

def fit_zoom_to_selected_nodes():
    nodegraphqt.fit_to_selection()

def show_undo_list():
    nodegraphqt._clear_actions()

def set_pipe_style(style):
    nodegraphqt.set_pipe_layout(style)

def set_background_grid(grid):
    nodegraphqt.set_pipe_layout(grid)

def auto_layout_downstream():
    nodegraphqt.auto_layout_nodes("downstream")

def auto_layout_upstream():
    nodegraphqt.auto_layout_nodes("upstream")

def toggle_node_search():
    nodegraphqt.clear_bin()
```