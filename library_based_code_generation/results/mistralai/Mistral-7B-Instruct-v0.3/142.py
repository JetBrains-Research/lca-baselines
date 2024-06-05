 Here is a basic implementation of a text editor using the urwid library in Python. This code provides the functionality you requested, but it's a simplified version and may need further refinement.

```python
import urwid

class LineEditor(urwid.Edit):
    def __init__(self, line_number, text, focus_set=True):
        super().__init__(text, minlen=1, multiline=True)
        self.line_number = line_number

    def keypress(self, size, key):
        if key == 'backspace' and self.beginning_of_line():
            return urwid.Edit.keypress(self, size, 'delete')
        elif key == 'delete' and self.end_of_line():
            return urwid.Edit.keypress(self, size, 'backspace')
        elif key == 'enter':
            self.insert('\n')
        return super().keypress(size, key)

class CustomListWalker(urwid.ListWalker):
    def __init__(self, lines, focus_position=0):
        self.lines = lines
        self.focus_position = focus_position

    def item_at_index(self, index):
        return self.lines[index]

    def focus_position(self):
        return self.focus_position

    def set_focus_position(self, position):
        self.focus_position = position

class TextEditor(urwid.Pile):
    def __init__(self, filename):
        self.lines = self.lazy_load_file(filename)
        self.focus_position = 0

        self.footer = urwid.AttrMap(urwid.Text("Press 's' to save, 'q' to quit, arrow keys to navigate"), 'footer')
        self.list_box = urwid.ListBox(self.CustomListWalker(self.lines, self.focus_position))

        self.edit_widget = LineEditor(self.focus_position, self.lines[self.focus_position])

        self.palette = [
            ('footer', 'standout', 'black', 'white'),
        ]

        self.add_widget(self.list_box)
        self.add_widget(self.edit_widget)
        self.add_widget(self.footer)

        self.connect_signal('focus_change', self.on_focus_change)

    def lazy_load_file(self, filename):
        # Implement lazy loading of the file here
        pass

    def on_focus_change(self, focus):
        if focus is self.edit_widget:
            line = self.lines[self.focus_position]
            self.edit_widget.set_text(line)
            self.focus_position = self.list_box.focus_position()
        else:
            self.edit_widget.set_text(self.lines[self.focus_position])

    def save(self, filename):
        # Implement saving the edited text back to the original file here
        pass

def main(filename):
    text_editor = TextEditor(filename)
    urwid.MainLoop(text_editor, unicode_input=True).run()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
```

This code creates a text editor with the specified functionality. The `TextEditor` class is the main class that handles the text editor's behavior, and the `lazy_load_file` and `save` methods should be implemented to load and save the text file as needed. The `main` function initializes the text editor with a given filename and starts the main loop of the application.