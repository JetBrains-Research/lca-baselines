  Here is the code that defines the functions to manipulate a node graph using the nodegraphqt library:
```
import nodegraphqt

def zoom_in():
    nodegraphqt.set_zoom(nodegraphqt.get_zoom() + 0.1)

def zoom_out():
    nodegraphqt.set_zoom(nodegraphqt.get_zoom() - 0.1)

def reset_zoom():
    nodegraphqt.set_zoom(1.0)

def set_layout_direction(direction):
    nodegraphqt.set_layout_direction(direction)

def open():
    nodegraphqt.open()

def import():
    nodegraphqt.import()

def save():
    nodegraphqt.save()

def clear_session():
    nodegraphqt.clear_session()

def clear_undo_history():
    nodegraphqt.clear_undo_stack()

def copy():
    nodegraphqt.copy()

def cut():
    nodegraphqt.cut()

def paste():
    nodegraphqt.paste()

def delete():
    nodegraphqt.delete()

def extract():
    nodegraphqt.extract()

def clear_connections():
    nodegraphqt.clear_connections()

def select_all():
    nodegraphqt.select_all()

def clear_selection():
    nodegraphqt.clear_selection()

def invert_selection():
    nodegraphqt.invert_selection()

def disable():
    nodegraphqt.disable()

def duplicate():
    nodegraphqt.duplicate()

def expand_group_of_nodes():
    nodegraphqt.expand_group_of_nodes()

def fit_zoom_level_to_selected_nodes():
    nodegraphqt.fit_to_selection()

def show_undo_list():
    nodegraphqt.show_undo_list()

def set_pipe_style_to_curved():
    nodegraphqt.set_pipe_layout(nodegraphqt.PipeLayout.CURVED)

def set_pipe_style_to_straight():
    nodegraphqt.set_pipe_layout(nodegraphqt.PipeLayout.STRAIGHT)

def set_pipe_style_to_angled():
    nodegraphqt.set_pipe_layout(nodegraphqt.PipeLayout.ANGLED)

def set_background_grid_to_none():
    nodegraphqt.set_background_grid(nodegraphqt.BackgroundGrid.NONE)

def set_background_grid_to_dots():
    nodegraphqt.set_background_grid(nodegraphqt.BackgroundGrid.DOTS)

def set_background_grid_to_lines():
    nodegraphqt.set_background_grid(nodegraphqt.BackgroundGrid.LINES)

def auto_layout_nodes_downstream():
    nodegraphqt.auto_layout_nodes(nodegraphqt.AutoLayout.DOWNSTREAM)

def auto_layout_nodes_upstream():
    nodegraphqt.auto_layout_nodes(nodegraphqt.AutoLayout.UPSTREAM)

def toggle_node_search():
    nodegraphqt.toggle_node_search()
```