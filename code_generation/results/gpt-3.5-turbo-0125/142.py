import urwid

class TextEditor:
    def __init__(self, filename):
        self.filename = filename
        self.text = self.load_text()
        self.list_walker = urwid.SimpleFocusListWalker(self.text)
        self.list_box = urwid.ListBox(self.list_walker)
        self.footer = urwid.Text("Instructions: Save - Ctrl+S, Quit - Ctrl+Q, Delete - Backspace, New Line - Enter, Navigate - Arrow keys")
        self.layout = urwid.Frame(body=self.list_box, footer=self.footer)
        self.loop = urwid.MainLoop(self.layout, unhandled_input=self.handle_keypress)
    
    def load_text(self):
        with open(self.filename, 'r') as file:
            return [urwid.Text(line) for line in file.readlines()]
    
    def save_text(self):
        with open(self.filename, 'w') as file:
            for item in self.list_walker:
                file.write(item.text + '\n')
    
    def handle_keypress(self, key):
        if key in ('ctrl q', 'ctrl Q'):
            raise urwid.ExitMainLoop()
        elif key in ('ctrl s', 'ctrl S'):
            self.save_text()
        elif key == 'backspace':
            # Handle deleting at the end of lines
            pass
        elif key == 'enter':
            # Handle starting new lines
            pass
        elif key == 'left':
            # Handle navigating left
            pass
        elif key == 'right':
            # Handle navigating right
            pass

def main(filename):
    editor = TextEditor(filename)
    editor.loop.run()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python text_editor.py <filename>")
    else:
        main(sys.argv[1])