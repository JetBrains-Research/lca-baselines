  import urwid

class DisplaySettingsMenu(urwid.Menu):
    def __init__(self, display):
        self.display = display
        self.brightness_slider = urwid.Slider(0, 100, 50, 1, self.on_brightness_change)
        self.contrast_slider = urwid.Slider(0, 100, 50, 1, self.on_contrast_change)
        self.cursor_style_menu = urwid.Menu([
            urwid.MenuItem('Solid', self.on_cursor_style_change, self.display.cursor_style == 'solid'),
            urwid.MenuItem('Underline', self.on_cursor_style_change, self.display.cursor_style == 'underline'),
            urwid.MenuItem('Block', self.on_cursor_style_change, self.display.cursor_style == 'block')
        ])
        self.led_color_menu = urwid.Menu([
            urwid.MenuItem('Red', self.on_led_color_change, self.display.led_color == 'red'),
            urwid.MenuItem('Green', self.on_led_color_change, self.display.led_color == 'green'),
            urwid.MenuItem('Blue', self.on_led_color_change, self.display.led_color == 'blue')
        ])
        self.about_menu = urwid.Menu([
            urwid.MenuItem('About this Demo', self.on_about_menu_change)
        ])
        super().__init__([
            urwid.MenuItem('Display Settings', self.display_settings_menu),
            urwid.MenuItem('Cursor Settings', self.cursor_settings_menu),
            urwid.MenuItem('LED Settings', self.led_settings_menu),
            urwid.MenuItem('About this Demo', self.about_menu)
        ])

    def on_brightness_change(self, value):
        self.display.brightness = value

    def on_contrast_change(self, value):
        self.display.contrast = value

    def on_cursor_style_change(self, value):
        self.display.cursor_style = value

    def on_led_color_change(self, value):
        self.display.led_color = value

    def on_about_menu_change(self):
        pass

class CursorSettingsMenu(urwid.Menu):
    def __init__(self, display):
        self.display = display
        self.cursor_style_menu = urwid.Menu([
            urwid.MenuItem('Solid', self.on_cursor_style_change, self.display.cursor_style == 'solid'),
            urwid.MenuItem('Underline', self.on_cursor_style_change, self.display.cursor_style == 'underline'),
            urwid.MenuItem('Block', self.on_cursor_style_change, self.display.cursor_style == 'block')
        ])
        super().__init__([
            urwid.MenuItem('Cursor Settings', self.cursor_settings_menu)
        ])

    def on_cursor_style_change(self, value):
        self.display.cursor_style = value

class LEDSettingsMenu(urwid.Menu):
    def __init__(self, display):
        self.display = display
        self.led_color_menu = urwid.Menu([
            urwid.MenuItem('Red', self.on_led_color_change, self.display.led_color == 'red'),
            urwid.MenuItem('Green', self.on_led_color_change, self.display.led_color == 'green'),
            urwid.MenuItem('Blue', self.on_led_color_change, self.display.led_color == 'blue')
        ])
        super().__init__([
            urwid.MenuItem('LED Settings', self.led_settings_menu)
        ])

    def on_led_color_change(self, value):
        self.display.led_color = value

class AboutMenu(urwid.Menu):
    def __init__(self):
        super().__init__([
            urwid.MenuItem('About this Demo', self.on_about_menu_change)
        ])

    def on_about_menu_change(self):
        pass

class Display:
    def __init__(self, display):
        self.display = display
        self.brightness = 50
        self.contrast = 50
        self.cursor_style = 'solid'
        self.led_color = 'red'

    def update(self):
        self.display.brightness = self.brightness
        self.display.contrast = self.contrast
        self.display.cursor_style = self.cursor_style
        self.display.led_color = self.led_color

class App:
    def __init__(self, display):
        self.display = display
        self.display_settings_menu = DisplaySettingsMenu(self.display)
        self.cursor_settings_menu = CursorSettingsMenu(self.display)
        self.led_settings_menu = LEDSettingsMenu(self.display)
        self.about_menu = AboutMenu()
        self.menu = urwid.Menu([
            urwid.MenuItem('Display Settings', self.display_settings_menu),
            urwid.MenuItem('Cursor Settings', self.cursor_settings_menu),
            urwid.MenuItem('LED Settings', self.led_settings_menu),
            urwid.MenuItem('About this Demo', self.about_menu)
        ])
        self.main_loop = urwid.MainLoop(self.menu, unhandled_input=self.on_input)

    def on_input(self, key):
        if key == 'q':
            raise urwid.ExitMainLoop()
        elif key == 'esc':
            self.menu.focus_position = 0
        elif key == 'up':
            self.menu.focus_position -= 1
        elif key == 'down':
            self.menu.focus_position += 1
        elif key == 'enter':
            self.menu.focus_position = 0

    def run(self):
        self.main_loop.run()

if __name__ == '__main__':
    display = Display(urwid.raw_display.Screen())
    app = App(display)
    app.run()