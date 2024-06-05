```python
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors import FocusBehavior, CompoundSelectionBehavior

class SelectableGrid(FocusBehavior, CompoundSelectionBehavior, GridLayout):
    def __init__(self, **kwargs):
        super(SelectableGrid, self).__init__(**kwargs)
        self.cols = 8  # Example column count, adjust as needed

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        # Handle key down events
        pass

    def keyboard_on_key_up(self, window, keycode):
        # Handle key up events
        pass

    def on_touch_down(self, touch):
        # Handle touch events
        super(SelectableGrid, self).on_touch_down(touch)
        for child in self.children:
            if child.collide_point(*touch.pos):
                self.select_with_touch(child.index, touch)

    def select_node(self, node):
        node.background_color = (1, 0, 0, 1)  # Change background color to red
        print("Selected nodes:", self.selected_nodes)
        super(SelectableGrid, self).select_node(node)

    def deselect_node(self, node):
        node.background_color = (1, 1, 1, 1)  # Change background color back to default
        super(SelectableGrid, self).deselect_node(node)

    def add_node(self, node):
        self.add_widget(node)

class SelectableGridApp(App):
    def build(self):
        grid = SelectableGrid()
        for i in range(40):  # Create and add 40 buttons
            btn = Button(text=str(i + 1), size_hint=(None, None), size=(100, 100))
            btn.bind(on_touch_down=grid.on_touch_down)
            grid.add_node(btn)
        return grid

if __name__ == '__main__':
    SelectableGridApp().run()
```