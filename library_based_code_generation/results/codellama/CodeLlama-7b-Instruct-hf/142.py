  from urwid import *

class TextEditor(object):
    def __init__(self, filename):
        self.filename = filename
        self.text = self.load_text()
        self.list_walker = self.create_list_walker()
        self.display = self.create_display()

    def load_text(self):
        with open(self.filename, 'r') as f:
            text = f.read()
        return text

    def create_list_walker(self):
        return ListWalker(self.text)

    def create_display(self):
        list_box = ListBox(self.list_walker)
        footer = Text('Press q to quit, s to save, and h to go to the beginning of the line')
        return Panel(list_box, footer)

    def handle_keypress(self, key):
        if key == 's':
            self.save_text()
        elif key == 'q':
            raise urwid.ExitMainLoop()
        elif key == 'h':
            self.move_cursor_to_beginning_of_line()
        elif key == 'l':
            self.move_cursor_to_end_of_line()
        elif key == 'j':
            self.move_cursor_down()
        elif key == 'k':
            self.move_cursor_up()
        elif key == 'd':
            self.delete_character()
        elif key == 'b':
            self.backspace_character()
        elif key == 'n':
            self.start_new_line()
        elif key == 'm':
            self.combine_lines()
        elif key == 'p':
            self.split_lines()

    def save_text(self):
        with open(self.filename, 'w') as f:
            f.write(self.text)

    def move_cursor_to_beginning_of_line(self):
        pass

    def move_cursor_to_end_of_line(self):
        pass

    def move_cursor_down(self):
        pass

    def move_cursor_up(self):
        pass

    def delete_character(self):
        pass

    def backspace_character(self):
        pass

    def start_new_line(self):
        pass

    def combine_lines(self):
        pass

    def split_lines(self):
        pass

def main(filename):
    text_editor = TextEditor(filename)
    loop = urwid.MainLoop(text_editor.display, unhandled_input=text_editor.handle_keypress)
    loop.run()

if __name__ == '__main__':
    main(sys.argv[1])