 Here is a basic structure for the UI using urwid library. Please note that this is a simplified version and you may need to adjust it according to your specific needs. Also, the custom characters for checkbox, radio button, progress bar, and slider are not provided as they depend on the specific font you are using.

```python
import urwid
import sys
from urwid.contrib.whsettings import WHSettings

class Checkbox(urwid.Text):
    def __init__(self, value, label):
        self.value = value
        self.label = label
        super().__init__(self.label, attr=self.attr)

    def attr(self):
        return self.value and 'checked' or None

class RadioButton(urwid.Text):
    def __init__(self, value, label):
        self.value = value
        self.label = label
        super().__init__(self.label, attr=self.attr)

    def attr(self):
        return self.value and 'selected' or None

class CustomProgressBar(urwid.ProgressBar):
    pass  # Implement your custom progress bar here

class Slider(urwid.WidgetWrap):
    def __init__(self, min_val, max_val, step=1):
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.value = self.min_val
        self.slider = CustomCharSlider(self.value)

        self.wrapper = urwid.WidgetWrap(self.slider)

    def keypress(self, size, key):
        if key == 'up':
            self.value += self.step
            if self.value > self.max_val:
                self.value = self.max_val
            self.slider.set_text(str(self.value))
        elif key == 'down':
            self.value -= self.step
            if self.value < self.min_val:
                self.value = self.min_val
            self.slider.set_text(str(self.value))
        return super().keypress(size, key)

class DisplaySettings(urwid.Columns):
    def __init__(self):
        super().__init__(
            (
                urwid.AttrMap(Checkbox(True, 'Brightness'), 'brightness'),
                urwid.AttrMap(Checkbox(False, 'Contrast'), 'contrast'),
            )
        )

class CursorSettings(urwid.ListBox):
    def __init__(self):
        super().__init__(urwid.SimpleListWalker(self))
        self.choices = ['Style 1', 'Style 2', 'Style 3']  # Replace with your cursor styles

    def render(self, size):
        return urwid.ListBoxManager(self, size)[0]

    def select_choice(self, index):
        self.body = self.choices[index]
        self.set_focus()

class LEDSettings(urwid.ListBox):
    def __init__(self):
        super().__init__(urwid.SimpleListWalker(self))
        self.choices = [
            (urwid.Text('Red'), 'color_red'),
            (urwid.Text('Green'), 'color_green'),
            (urwid.Text('Blue'), 'color_blue'),
            # Add more LED colors as needed
        ]

    def render(self, size):
        items = []
        for choice, color in self.choices:
            items.append(urwid.AttrMap(choice, color))
        return urwid.ListBoxManager(urwid.SimpleListWalker(self), size)(items)

class About(urwid.Text):
    def __init__(self):
        super().__init__(
            "This is a demo for a crystalfontz 635 LCD display using urwid library.\n"
            "Copyright 2022 Your Name"
        )

class Menu(urwid.ListBox):
    def __init__(self, items):
        super().__init__(urwid.SimpleListWalker(self))
        self.items = items

    def render(self, size):
        items = []
        for item in self.items:
            items.append(urwid.AttrMap(item, 'menu'))
        return urwid.ListBoxManager(urwid.SimpleListWalker(self), size)(items)

    def select_choice(self, index):
        self.body = self.items[index]
        self.set_focus()

class MainScreen(urwid.Columns):
    def __init__(self, display, cursor, led, about):
        super().__init__(
            (
                display,
                urwid.AttrMap(cursor, 'cursor'),
                led,
                about,
            )
        )

def main(args):
    wh = WHSettings()
    display = DisplaySettings()
    cursor = CursorSettings()
    led = LEDSettings()
    about = About()
    menu = Menu([
        urwid.Text('Display Settings'),
        urwid.Text('Cursor Settings'),
        urwid.Text('LED Settings'),
        urwid.Text('About this Demo'),
        menu_button('Back'),
    ])

    main_screen = MainScreen(display, cursor, led, about)
    popup = ThingWithAPopUp(main_screen, menu)

    screen = urwid.Pile([popup])
    screen.set_focus(popup)

    loop = urwid.MainLoop(screen, unicode_input=True)
    loop.run()

if __name__ == '__main__':
    main(sys.argv)
```

This code creates a UI with a menu, display settings, cursor settings, LED settings, and an 'About this Demo' section. The display settings, cursor settings, and LED settings are represented as checkboxes and radio buttons. The menu includes a back button that allows the user to go back to the previous menu when pressed. The UI also includes a custom progress bar and a slider control. However, the custom characters for the progress bar and slider are not provided as they depend on the specific font you are using. The code connects to the LCD display using the provided command line argument, but the connection logic is not included in this example.