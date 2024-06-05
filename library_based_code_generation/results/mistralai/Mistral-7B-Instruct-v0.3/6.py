 Here is a basic structure for the UI using urwid library. Please note that this is a simplified version and you may need to adjust it according to your specific needs. Also, the custom characters for checkbox, radio button, progress bar, and slider are not provided as they depend on the specific font used by the crystalfontz 635 LCD display.

```python
import urwid

class CheckBox(urwid.WidgetWrap):
    def __init__(self, value, on_change):
        self.value = value
        self.on_change = on_change
        self.widget = urwid.Text(u'[]')
        self.connect_signal('change', self.on_change)

    def keypress(self, size, key):
        if key == 'space':
            self.value = not self.value
            self.widget.set_text(self.value and u'[X]' or u'[ ]')
            self.on_change(self.value)
        return super().keypress(size, key)

class RadioButton(urwid.WidgetWrap):
    # Similar structure as CheckBox for RadioButton
    pass

class ProgressBar(urwid.WidgetWrap):
    # Similar structure as CheckBox for ProgressBar
    pass

class Slider(urwid.WidgetWrap):
    def __init__(self, min_value, max_value, value, on_change):
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.on_change = on_change
        self.widget = urwid.Edit(u'[]', min_length=1, max_length=3)
        self.connect_signal('change', self.on_change)

    def keypress(self, size, key):
        if key == 'left':
            self.value = max(self.value - 1, self.min_value)
            self.widget.set_text(u'[' + self.value * '>' + (self.max_value - self.value) * ' ' + ']')
            self.on_change(self.value)
        elif key == 'right':
            self.value = min(self.value + 1, self.max_value)
            self.widget.set_text(u'[' + self.value * '>' + (self.max_value - self.value) * ' ' + ']')
            self.on_change(self.value)
        return super().keypress(size, key)

def make_menu(parent, title, options):
    menu = urwid.ListBox(urwid.SimpleListWalker(urwid.Padding(urwid.Column([urwid.Text(title), *options]), left=2, right=2))),
    menu.divider = urwid.Divider(u'-' * len(title))
    parent.append(menu)
    return menu

def main(lcd_connection):
    # Initialize the main window
    main_window = urwid.Columns([
        urwid.AttrMap(urwid.Text('Display Settings'), 'header'),
        urwid.AttrMap(urwid.ListBox(urwid.SimpleListWalker(options1)), 'options'),

        urwid.AttrMap(urwid.Text('Cursor Settings'), 'header'),
        urwid.AttrMap(urwid.ListBox(urwid.SimpleListWalker(options2)), 'options'),

        urwid.AttrMap(urwid.Text('LED Settings'), 'header'),
        urwid.AttrMap(urwid.ListBox(urwid.SimpleListWalker(options3)), 'options'),

        urwid.AttrMap(urwid.Text('About this Demo'), 'header'),
        urwid.AttrMap(urwid.Text('Text about the demo'), 'body'),
    ])

    # Connect to the LCD display
    lcd_connection.connect()

    # Run the application
    loop = urwid.MainLoop(main_window, screen_width=lcd_connection.get_width(), screen_height=lcd_connection.get_height())
    loop.run()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lcd_port', help='LCD port')
    args = parser.parse_args()
    from crystalfontz_lcd import CrystalFontzLCD
    lcd = CrystalFontzLCD(args.lcd_port)
    main(lcd)
```

This code creates a simple UI with different sections for display settings, cursor settings, LED settings, and an 'About this Demo' section. Each section is represented as a list of options. You will need to fill in the `options1`, `options2`, and `options3` variables with the actual options for each section. The `make_menu` function is used to create the menus. The main loop runs the application and connects to the LCD display using the provided command line argument.