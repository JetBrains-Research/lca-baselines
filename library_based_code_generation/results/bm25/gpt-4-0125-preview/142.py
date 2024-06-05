```python
import urwid

class LazyFileLoader(urwid.ListWalker):
    def __init__(self, filename):
        self.file = open(filename, 'r+')
        self.lines = []
        self.focus = 0

    def get_focus(self):
        return self._get_at_pos(self.focus)

    def set_focus(self, focus):
        self.focus = focus
        self._modified()

    def get_next(self, start_from):
        return self._get_at_pos(start_from + 1)

    def get_prev(self, start_from):
        return self._get_at_pos(start_from - 1)

    def _get_at_pos(self, pos):
        if pos < 0:
            return None, None
        if len(self.lines) <= pos:
            self._load_next_line()
        try:
            return urwid.Text(self.lines[pos]), pos
        except IndexError:
            return None, None

    def _load_next_line(self):
        line = self.file.readline()
        if line:
            self.lines.append(line.rstrip("\n"))

    def save_file(self):
        self.file.seek(0)
        self.file.writelines(line + '\n' for line in self.lines)
        self.file.truncate()
        self.file.flush()

class TextEditor:
    def __init__(self, filename):
        self.loader = LazyFileLoader(filename)
        self.listbox = urwid.ListBox(self.loader)
        self.footer = urwid.Text("Commands: Save (Ctrl+S), Quit (Ctrl+Q)")
        self.view = urwid.Frame(header=None, body=self.listbox, footer=self.footer)

    def main(self):
        loop = urwid.MainLoop(self.view, unhandled_input=self.handle_input)
        loop.run()

    def handle_input(self, key):
        if key in ('ctrl q', 'Q'):
            raise urwid.ExitMainLoop()
        elif key == 'ctrl s':
            self.loader.save_file()
        elif key == 'enter':
            self.insert_line()
        elif key in ('backspace', 'delete'):
            self.delete_line()

    def insert_line(self):
        focus_widget, idx = self.loader.get_focus()
        if focus_widget:
            text = focus_widget.get_text()[0]
            self.loader.lines.insert(idx + 1, "")
            self.loader.set_focus(idx + 1)

    def delete_line(self):
        focus_widget, idx = self.loader.get_focus()
        if focus_widget and idx > 0:
            del self.loader.lines[idx]
            self.loader.set_focus(idx - 1)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: text_editor.py filename")
        sys.exit(1)
    filename = sys.argv[1]
    editor = TextEditor(filename)
    editor.main()
```