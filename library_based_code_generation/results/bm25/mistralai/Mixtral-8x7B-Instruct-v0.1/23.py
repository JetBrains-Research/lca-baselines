from kivy.uix.behaviors import FocusBehavior, CompoundSelectionBehavior
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button

class SelectableGrid(FocusBehavior, CompoundSelectionBehavior, GridLayout):
def __init__(self, **kwargs):
super().__init__(**kwargs)
self.selected_nodes = set()
for i in range(40):
btn = Button(text=str(i + 1), size_hint_y=None, height=50)
self.add_widget(btn)
btn.bind(on_touch_down=self._touch_down)
self.bind(size=self.update_selection_rect, pos=self.update_selection_rect)

def update_selection_rect(self, *args):
if not self.children:
return

x, y = self.to_widget(*self.pos)
width, height = self.width, self.height
self.selection_rect = (x, y, width, height)

def select_with_key_down(self, key, *largs):
if key == 'down':
current_row, current_col = self.get_index_of_node()
if current_col < self.cols - 1:
self.select_node(current_row, current_col + 1)
elif current_row < self.rows - 1:
self.select_node(current_row + 1, 0)
self.print_selected_nodes()

def select_with_key_up(self, key, *largs):
if key == 'up':
current_row, current_col = self.get_index_of_node()
if current_col > 0:
self.select_node(current_row, current_col - 1)
elif current_row > 0:
self.select_node(current_row - 1, self.cols - 1)
self.print_selected_nodes()

def select_with_touch(self, touch):
if self.collide_point(*touch.pos) and self.focus:
node_index = self.get_node_index(touch)
self.select_node(*node_index)
self.print_selected_nodes()

def get_node_index(self, touch):
for i, child in enumerate(self.children):
if child.collide_point(*touch.pos):
return divmod(i, self.cols)
return -1, -1

def select_node(self, row, col):
node = self.children[row * self.cols + col]
if node in self.selected_nodes:
self.deselect_node(node)
else:
self.select_node(node)

def deselect_node(self, node):
node.background_normal = ''
node.background_color = [1, 1, 1, 1]
self.selected_nodes.discard(node)

def select_node(self, node):
node.background_normal = 'atlas://data/images/defaulttheme/button'
node.background_color = [0, 0.5, 1, 1]
self.selected_nodes.add(node)

def print_selected_nodes(self):
print("Selected nodes:", [node.text for node in self.selected_nodes])

if __name__ == '__main__':
from kivy.app import App

grid = SelectableGrid(rows=4, cols=10, size_hint=(None, None), size=(400, 400))

class MyApp(App):
def build(self):
return grid

if __name__ == '__main__':
MyApp().run()