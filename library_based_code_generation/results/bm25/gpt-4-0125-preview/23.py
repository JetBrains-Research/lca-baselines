```python
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors import FocusBehavior, CompoundSelectionBehavior

class SelectableGrid(FocusBehavior, CompoundSelectionBehavior, GridLayout):
    def __init__(self, **kwargs):
        super(SelectableGrid, self).__init__(**kwargs)
        self.cols = 8
        self.bind(selected_nodes=self.on_selection_change)

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        # Handle key down events
        pass

    def keyboard_on_key_up(self, window, keycode):
        # Handle key up events
        pass

    def navigate_to_node(self, node_number):
        # Navigate to a node by typing its number
        pass

    def select_node(self, node):
        # Select a node
        node.background_color = (1, 0, 0, 1)  # Change background color to red
        self.select_with_touch(node)

    def deselect_node(self, node):
        # Deselect a node
        node.background_color = (1, 1, 1, 1)  # Change background color to white
        self.deselect_node(node)

    def on_touch_down(self, touch):
        # Handle touch events
        if super(SelectableGrid, self).on_touch_down(touch):
            return True
        return False

    def on_selection_change(self, instance, value):
        # Handle selection changes
        print("Selected nodes:", self.selected_nodes)

class SelectableGridApp(App):
    def build(self):
        grid = SelectableGrid()
        for i in range(40):
            btn = Button(text=str(i + 1))
            btn.bind(on_press=grid.select_node)
            grid.add_widget(btn)
        return grid

if __name__ == '__main__':
    SelectableGridApp().run()
```