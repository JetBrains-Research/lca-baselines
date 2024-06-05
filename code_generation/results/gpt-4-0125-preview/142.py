```python
import urwid

class LazyFileLoader(urwid.ListWalker):
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'r')
        self.lines = []
        self.focus = 0

    def get_focus(self):
        return self._get_line(self.focus)

    def set_focus(self, focus):
        self.focus = focus
        self._modified()

    def get_next(self, start_from):
        return self._get_line(start_from + 1)

    def get_prev(self, start_from):
        return self._get_line(start_from - 1)

    def _get_line(self, index):
        while index >= len(self.lines) and not self.file.closed:
            line = self.file.readline()
            if not line:
                self.file.close()
                break
            self.lines.append(urwid.Text(line.rstrip('\n')))
        if index < len(self.lines):
            return self.lines[index], index
        else:
            return None, None

class TextEditor:
    def __init__(self, filename):
        self.filename = filename
        self.loader = LazyFileLoader(filename)
        self.listbox = urwid.ListBox(self.loader)
        self.footer = urwid.Text("CTRL+S: Save, CTRL+Q: Quit, ARROWS: Navigate, BACKSPACE: Delete, ENTER: New Line")
        self.view = urwid.Frame(header=None, body=self.listbox, footer=self.footer)

    def main(self):
        loop = urwid.MainLoop(self.view, unhandled_input=self.handle_input)
        loop.run()

    def handle_input(self, key):
        if key == 'ctrl s':
            self.save_file()
        elif key == 'ctrl q':
            raise urwid.ExitMainLoop()
        elif key == 'enter':
            self.insert_line()
        elif key == 'backspace':
            self.delete_character()
        elif key in ('left', 'right', 'up', 'down'):
            self.navigate(key)

    def save_file(self):
        with open(self.filename, 'w') as f:
            for widget, _ in self.loader:
                f.write(widget.text + '\n')

    def insert_line(self):
        focus_widget, idx = self.loader.get_focus()
        if focus_widget:
            self.loader.lines.insert(idx + 1, urwid.Text(''))
            self.loader.set_focus(idx + 1)

    def delete_character(self):
        focus_widget, idx = self.loader.get_focus()
        if focus_widget:
            text = focus_widget.text
            if len(text) > 0:
                focus_widget.set_text(text[:-1])
            elif idx > 0:
                del self.loader.lines[idx]
                self.loader.set_focus(idx - 1)

    def navigate(self, direction):
        focus_widget, idx = self.loader.get_focus()
        if direction == 'left':
            # Move cursor left or to the end of the previous line if at the beginning
            pass  # Implement cursor movement within a line if desired
        elif direction == 'right':
            # Move cursor right or to the beginning of the next line if at the end
            pass  # Implement cursor movement within a line if desired
        elif direction == 'up' and idx > 0:
            self.loader.set_focus(idx - 1)
        elif direction == 'down' and idx < len(self.loader.lines) - 1:
            self.loader.set_focus(idx + 1)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: text_editor.py filename")
        sys.exit(1)
    filename = sys.argv[1]
    editor = TextEditor(filename)
    editor.main()
```