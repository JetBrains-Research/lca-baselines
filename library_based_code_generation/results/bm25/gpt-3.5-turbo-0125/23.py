```python
from kivy.uix.behaviors import FocusBehavior, CompoundSelectionBehavior
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.app import App

class SelectableGrid(FocusBehavior, CompoundSelectionBehavior, GridLayout):
    def __init__(self, **kwargs):
        super(SelectableGrid, self).__init__(**kwargs)

    def on_selected_nodes(self, nodes):
        print("Selected nodes:", nodes)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            return self.select_with_touch(touch)

    def on_touch_up(self, touch):
        if self.selected_nodes:
            self.deselect_node(self.selected_nodes[0])

    def on_key_down(self, key, modifiers):
        if key == 'up':
            self.select_with_key_down()
        elif key == 'down':
            self.select_with_key_up()

    def on_key_up(self, key):
        if key == 'enter':
            self.navigate_to_node(int(self.selected_nodes[0].text))

grid = SelectableGrid(cols=5, rows=8)
for i in range(1, 41):
    button = Button(text=str(i))
    button.bind(on_touch_down=grid.on_touch_down)
    grid.add_widget(button)

class TestApp(App):
    def build(self):
        return grid

if __name__ == '__main__':
    TestApp().run()
```