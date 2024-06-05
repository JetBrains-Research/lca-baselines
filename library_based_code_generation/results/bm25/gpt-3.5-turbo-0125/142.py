import urwid

class LazyTextLoader(urwid.ListWalker):
    def __init__(self, filename):
        self.filename = filename
        self.lines = []
        with open(filename, 'r') as file:
            for line in file:
                self.lines.append(urwid.Text(line.rstrip('\n')))

    def get_focus(self):
        return self.lines[self.focus]

    def set_focus(self, focus):
        self.focus = focus

    def get_next(self, start_from):
        if start_from + 1 < len(self.lines):
            return (self.lines[start_from + 1], start_from + 1)
        else:
            return None, None

    def get_prev(self, start_from):
        if start_from > 0:
            return (self.lines[start_from - 1], start_from - 1)
        else:
            return None, None

class TextEditor(urwid.WidgetWrap):
    def __init__(self, filename):
        self.filename = filename
        self.text_loader = LazyTextLoader(filename)
        self.list_box = urwid.ListBox(self.text_loader)
        self.footer = urwid.Text("Instructions: Save - Ctrl+S, Quit - Ctrl+Q, Delete - Backspace, New Line - Enter, Navigate - Arrow keys")
        self.main_pile = urwid.Pile([self.list_box, self.footer])
        super().__init__(self.main_pile)

def main(filename):
    text_editor = TextEditor(filename)
    urwid.MainLoop(text_editor).run()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as an argument.")