  import urwid

class DisplaySettingsMenu(urwid.WidgetWrap):
    def __init__(self, display):
        self.display = display
        self.brightness_slider = urwid.Slider(0, 100, self.display.brightness)
        self.contrast_slider = urwid.Slider(0, 100, self.display.contrast)
        self.brightness_label = urwid.Text("Brightness:")
        self.contrast_label = urwid.Text("Contrast:")
        self.brightness_button = urwid.Button("Apply")
        self.contrast_button = urwid.Button("Apply")
        self.cancel_button = urwid.Button("Cancel")
        self.menu = urwid.Columns([
            urwid.Padding(self.brightness_label, left=2, right=2),
            urwid.Padding(self.brightness_slider, left=2, right=2),
            urwid.Padding(self.contrast_label, left=2, right=2),
            urwid.Padding(self.contrast_slider, left=2, right=2),
            urwid.Padding(self.brightness_button, left=2, right=2),
            urwid.Padding(self.contrast_button, left=2, right=2),
            urwid.Padding(self.cancel_button, left=2, right=2)
        ])
        super().__init__(self.menu)

    def keypress(self, size, key):
        if key == "enter":
            self.display.brightness = self.brightness_slider.get_value()
            self.display.contrast = self.contrast_slider.get_value()
            self.display.update_display()
        elif key == "esc":
            self.display.menu.show_previous_menu()
        return key

class CursorSettingsMenu(urwid.WidgetWrap):
    def __init__(self, display):
        self.display = display
        self.cursor_style_button = urwid.Button("Cursor Style")
        self.cancel_button = urwid.Button("Cancel")
        self.menu = urwid.Columns([
            urwid.Padding(self.cursor_style_button, left=2, right=2),
            urwid.Padding(self.cancel_button, left=2, right=2)
        ])
        super().__init__(self.menu)

    def keypress(self, size, key):
        if key == "enter":
            self.display.menu.show_previous_menu()
        elif key == "esc":
            self.display.menu.show_previous_menu()
        return key

class LEDSettingsMenu(urwid.WidgetWrap):
    def __init__(self, display):
        self.display = display
        self.led_color_button = urwid.Button("LED Color")
        self.cancel_button = urwid.Button("Cancel")
        self.menu = urwid.Columns([
            urwid.Padding(self.led_color_button, left=2, right=2),
            urwid.Padding(self.cancel_button, left=2, right=2)
        ])
        super().__init__(self.menu)

    def keypress(self, size, key):
        if key == "enter":
            self.display.menu.show_previous_menu()
        elif key == "esc":
            self.display.menu.show_previous_menu()
        return key

class AboutDemoMenu(urwid.WidgetWrap):
    def __init__(self, display):
        self.display = display
        self.about_demo_label = urwid.Text("About this Demo")
        self.cancel_button = urwid.Button("Cancel")
        self.menu = urwid.Columns([
            urwid.Padding(self.about_demo_label, left=2, right=2),
            urwid.Padding(self.cancel_button, left=2, right=2)
        ])
        super().__init__(self.menu)

    def keypress(self, size, key):
        if key == "enter":
            self.display.menu.show_previous_menu()
        elif key == "esc":
            self.display.menu.show_previous_menu()
        return key

class Menu(urwid.WidgetWrap):
    def __init__(self, display):
        self.display = display
        self.display_settings_menu = DisplaySettingsMenu(self.display)
        self.cursor_settings_menu = CursorSettingsMenu(self.display)
        self.led_settings_menu = LEDSettingsMenu(self.display)
        self.about_demo_menu = AboutDemoMenu(self.display)
        self.menu_options = [
            ("Display Settings", self.display_settings_menu),
            ("Cursor Settings", self.cursor_settings_menu),
            ("LED Settings", self.led_settings_menu),
            ("About this Demo", self.about_demo_menu)
        ]
        self.menu_list = urwid.ListBox(urwid.SimpleFocusListWalker(self.menu_options))
        self.menu = urwid.Padding(self.menu_list, left=2, right=2)
        super().__init__(self.menu)

    def show_previous_menu(self):
        self.menu_list.set_focus(self.menu_list.get_focus() - 1)

    def keypress(self, size, key):
        if key == "enter":
            self.menu_list.set_focus(self.menu_list.get_focus() + 1)
        elif key == "esc":
            self.display.menu.show_previous_menu()
        return key

class Display(urwid.WidgetWrap):
    def __init__(self, display):
        self.display = display
        self.menu = Menu(self)
        self.text = urwid.Text("")
        self.progress_bar = urwid.ProgressBar(0, 100)
        self.checkbox = urwid.CheckBox("Checkbox")
        self.radio_button = urwid.RadioButton("Radio Button")
        self.horizontal_slider = urwid.Slider(0, 100, 50)
        self.menu_button = urwid.Button("Menu")
        self.widgets = [
            self.text,
            self.progress_bar,
            self.checkbox,
            self.radio_button,
            self.horizontal_slider,
            self.menu_button
        ]
        self.widget = urwid.Columns(self.widgets)
        super().__init__(self.widget)

    def update_display(self):
        self.text.set_text(self.display.text)
        self.progress_bar.set_completion(self.display.progress)
        self.checkbox.set_state(self.display.checkbox)
        self.radio_button.set_state(self.display.radio_button)
        self.horizontal_slider.set_value(self.display.horizontal_slider)

    def keypress(self, size, key):
        if key == "esc":
            self.menu.show_previous_menu()
        return key

def main():
    display = Display(None)
    urwid.MainLoop(display, unhandled_input=display.keypress).run()

if __name__ == "__main__":
    main()