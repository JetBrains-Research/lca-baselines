```python
import urwid

def menu_button(caption, callback):
    button = urwid.Button(caption)
    urwid.connect_signal(button, 'click', callback)
    return urwid.AttrMap(button, None, focus_map='reversed')

def menu(title, choices):
    body = [urwid.Text(title), urwid.Divider()]
    for c in choices:
        button = menu_button(c[0], c[1])
        body.append(urwid.AttrMap(button, None, focus_map='reversed'))
    return urwid.ListBox(urwid.SimpleFocusListWalker(body))

def display_settings():
    txt = urwid.Text("Display Settings")
    brightness = urwid.Slider("Brightness")
    contrast = urwid.Slider("Contrast")
    back = menu_button("Back", main_menu)
    items = [txt, brightness, contrast, back]
    content = urwid.ListBox(urwid.SimpleFocusListWalker(items))
    return urwid.BoxAdapter(content, height=20)

def cursor_settings():
    txt = urwid.Text("Cursor Settings")
    styles = ["Block", "Underline", "Bar"]
    radio_group = []
    for style in styles:
        button = urwid.RadioButton(radio_group, style)
        urwid.connect_signal(button, 'change', on_radio_change)
    back = menu_button("Back", main_menu)
    items = [txt] + radio_group + [back]
    content = urwid.ListBox(urwid.SimpleFocusListWalker(items))
    return urwid.BoxAdapter(content, height=20)

def led_settings():
    txt = urwid.Text("LED Settings")
    color_picker = urwid.Edit("Color: ")
    apply_btn = urwid.Button("Apply")
    back = menu_button("Back", main_menu)
    items = [txt, color_picker, apply_btn, back]
    content = urwid.ListBox(urwid.SimpleFocusListWalker(items))
    return urwid.BoxAdapter(content, height=20)

def about_demo():
    txt = urwid.Text("About this Demo\nThis is a demo of the Crystalfontz 635 LCD display interface.")
    back = menu_button("Back", main_menu)
    content = urwid.ListBox(urwid.SimpleFocusListWalker([txt, back]))
    return urwid.BoxAdapter(content, height=20)

def on_radio_change(button, new_state, user_data=None):
    if new_state:
        print("Selected:", button.label)

def main_menu(button=None):
    choices = [
        ("Display Settings", lambda button: main.original_widget.open_box(display_settings())),
        ("Cursor Settings", lambda button: main.original_widget.open_box(cursor_settings())),
        ("LED Settings", lambda button: main.original_widget.open_box(led_settings())),
        ("About this Demo", lambda button: main.original_widget.open_box(about_demo())),
    ]
    return menu("Main Menu", choices)

class CascadingBoxes(urwid.WidgetPlaceholder):
    max_box_levels = 4

    def __init__(self, box):
        super().__init__(urwid.SolidFill(u'\N{MEDIUM SHADE}'))
        self.box_level = 0
        self.open_box(box)

    def open_box(self, box):
        self.original_widget = urwid.Overlay(urwid.LineBox(box),
                                             self.original_widget,
                                             align='center', width=('relative', 80),
                                             valign='middle', height=('relative', 80),
                                             min_width=24, min_height=8,
                                             left=self.box_level * 3,
                                             right=(self.max_box_levels - self.box_level - 1) * 3,
                                             top=self.box_level * 2,
                                             bottom=(self.max_box_levels - self.box_level - 1) * 2)
        self.box_level = (self.box_level + 1) % self.max_box_levels

def exit_program(button):
    raise urwid.ExitMainLoop()

main = CascadingBoxes(main_menu())
urwid.MainLoop(main, palette=[('reversed', 'standout', '')]).run()
```