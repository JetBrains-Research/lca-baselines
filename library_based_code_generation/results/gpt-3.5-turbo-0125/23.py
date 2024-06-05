```python
from kivy.uix.behaviors import FocusBehavior, CompoundSelectionBehavior
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.app import App

class SelectableGrid(FocusBehavior, CompoundSelectionBehavior, GridLayout):
    def __init__(self, **kwargs):
        super(SelectableGrid, self).__init__(**kwargs)
    
    def on_selected_nodes(self, gird, nodes):
        print("Selected nodes:", nodes)
    
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            return self.select_with_touch(touch)
        return super(SelectableGrid, self).on_touch_down(touch)
    
    def on_touch_up(self, touch):
        if self.selected_nodes and not self.collide_point(*touch.pos):
            self.deselect_node()
        return super(SelectableGrid, self).on_touch_up(touch)

    def on_key_down(self, window, key, *args):
        # Handle key down event
        pass

    def on_key_up(self, window, key, *args):
        # Handle key up event
        pass

    def navigate_to_node(self, node_number):
        # Navigate to a node by typing its number
        pass

    def select_node(self, node):
        # Select a node
        pass

    def deselect_node(self):
        # Deselect a node
        pass

grid = SelectableGrid(cols=5, rows=8)
for i in range(1, 41):
    button = Button(text=str(i))
    button.bind(on_touch_down=grid.on_touch_down)
    grid.add_widget(button)

class TestApp(App):
    def build(self):
        return grid

TestApp().run()
```