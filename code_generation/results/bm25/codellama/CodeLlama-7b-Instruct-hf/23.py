  Here is the Python code for the custom class "SelectableGrid" that inherits from "FocusBehavior", "CompoundSelectionBehavior", and "GridLayout" from the Kivy library:
```
from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors import FocusBehavior, CompoundSelectionBehavior

class SelectableGrid(FocusBehavior, CompoundSelectionBehavior, GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(on_key_down=self.handle_key_down, on_key_up=self.handle_key_up)
        self.bind(on_touch_down=self.handle_touch_down)

    def handle_key_down(self, instance, keycode, text, modifiers):
        if keycode[1] == '1':
            self.select_with_key_down(1)
        elif keycode[1] == '2':
            self.select_with_key_down(2)
        elif keycode[1] == '3':
            self.select_with_key_down(3)
        elif keycode[1] == '4':
            self.select_with_key_down(4)
        elif keycode[1] == '5':
            self.select_with_key_down(5)
        elif keycode[1] == '6':
            self.select_with_key_down(6)
        elif keycode[1] == '7':
            self.select_with_key_down(7)
        elif keycode[1] == '8':
            self.select_with_key_down(8)
        elif keycode[1] == '9':
            self.select_with_key_down(9)
        elif keycode[1] == '0':
            self.select_with_key_down(0)

    def handle_key_up(self, instance, keycode, text, modifiers):
        if keycode[1] == '1':
            self.select_with_key_up(1)
        elif keycode[1] == '2':
            self.select_with_key_up(2)
        elif keycode[1] == '3':
            self.select_with_key_up(3)
        elif keycode[1] == '4':
            self.select_with_key_up(4)
        elif keycode[1] == '5':
            self.select_with_key_up(5)
        elif keycode[1] == '6':
            self.select_with_key_up(6)
        elif keycode[1] == '7':
            self.select_with_key_up(7)
        elif keycode[1] == '8':
            self.select_with_key_up(8)
        elif keycode[1] == '9':
            self.select_with_key_up(9)
        elif keycode[1] == '0':
            self.select_with_key_up(0)

    def handle_touch_down(self, instance, touch):
        node = self.get_index_of_node(touch.x, touch.y)
        if node:
            self.select_with_touch(node)

    def select_with_key_down(self, node):
        self.deselect_all()
        self.select_node(node)
        self.print_selected_nodes()

    def select_with_key_up(self, node):
        self.deselect_all()
        self.select_node(node)
        self.print_selected_nodes()

    def select_with_touch(self, node):
        self.deselect_all()
        self.select_node(node)
        self.print_selected_nodes()

    def print_selected_nodes(self):
        print(self.get_selected_nodes())

    def deselect_all(self):
        for node in self.get_selected_nodes():
            node.background_color = (0, 0, 0, 0)

    def select_node(self, node):
        node.background_color = (1, 0, 0, 1)

    def get_selected_nodes(self):
        return [node for node in self.children if node.background_color == (1, 0, 0, 1)]

    def get_index_of_node(self, x, y):
        for i, node in enumerate(self.children):
            if node.collide_point(x, y):
                return i
        return None
```
To create an instance of the "SelectableGrid" class with specific parameters and add 40 buttons to it, each with a unique number as its text, you can use the following code:
```
grid = SelectableGrid(cols=4, rows=10)
for i in range(40):
    button = Button(text=str(i), size_hint=(None, None), size=(100, 100))
    grid.add_widget(button)
    button.bind(on_touch_down=grid.handle_touch_down)
```
To bind the touch down event of each button to the touch handling method of the grid, you can use the following code:
```
for button in grid.children:
    button.bind(on_touch_down=grid.handle_touch_down)
```
Finally, to run the application with the grid as the root widget, you can use the following code:
```
from kivy.app import App
from kivy.uix.widget import Widget

class MyApp(App):
    def build(self):
        return grid

if __name__ == '__main__':
    MyApp().run()
```