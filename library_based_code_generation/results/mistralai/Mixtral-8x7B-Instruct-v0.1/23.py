from kivy.uix.behaviors import FocusBehavior
from kivy.uix.behaviors import CompoundSelectionBehavior
from kivy.uix.gridlayout import GridLayout

class SelectableGrid(FocusBehavior, CompoundSelectionBehavior, GridLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_nodes = set()
        self.bind(size=self.update_selection_rectangle)
        self.bind(pos=self.update_selection_rectangle)

    def update_selection_rectangle(self, instance, value):
        if not self.nodes or not all(node.collide_point(*value) for node in self.nodes):
            return
        for node in self.nodes:
            node.select_with_toggle()

    def on_key_down(self, instance, keycode, text, modifiers):
        if keycode[1] == 'numpad2':
            self.navigate_to_node(1)
        elif keycode[1] == 'numpad4':
            self.navigate_to_node(2)
        elif keycode[1] == 'numpad6':
            self.navigate_to_node(3)
        elif keycode[1] == 'numpad8':
            self.navigate_to_node(4)

    def navigate_to_node(self, node_number):
        for i, node in enumerate(self.children):
            if i == node_number - 1:
                node.focus = True
                break

    def on_touch_down(self, touch):
        if touch.grab_current is not self:
            for node in self.nodes:
                if node.collide_point(*touch.pos) and node.focus:
                    node.select_with_toggle()
                    print("Selected nodes:", [child.text for child in self.selection])
                    break

    def on_selected_nodes(self, instance, nodes):
        for node in nodes:
            node.background_color = [1, 0, 0, 1] if node in self.selected_nodes else [1, 1, 1, 1]

if __name__ == "__main__":
    from kivy.app import App

    class SelectableGridApp(App):
        def build(self):
            grid = SelectableGrid(cols=4, rows=10)
            for i in range(1, 41):
                button = Button(text=str(i), size_hint=(None, None), width=50, height=50)
                button.bind(on_touch_down=grid.on_touch_down)
                grid.add_widget(button)
            return grid

    SelectableGridApp().run()