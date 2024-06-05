 import urwid

class LazyLoader(urwid.Walker):
def __init__(self, file\_name):
self.file\_name = file\_name
self.file\_handle = open(file\_name, 'r')
self.position = 0

def get\_length(self):
return None

def get\_at(self, index):
if index < self.position or index >= self.position + 100:
self.position = index
self.file\_handle.seek(index)
return self.file\_handle.readline()

def go\_to(self, index):
self.position = index
self.file\_handle.seek(index)

class CustomListBox(urwid.ListBox):
def __init__(self, lazy\_loader):
self.lazy\_loader = lazy\_loader
urwid.ListBox.__init__(self, urwid.SimpleListWalker(self.lazy\_loader))

class Footer(urwid.WidgetWrap):
def __init__(self):
self.footer\_text = urwid.Text("(use arrow keys to navigate, 's' to save, 'q' to quit)")
urwid.WidgetWrap.__init__(self, self.footer\_text)

class MyTextEditor(urwid.Frame):
def __init__(self, file\_name):
lazy\_loader = LazyLoader(file\_name)
listbox = CustomListBox(lazy\_loader)
footer = Footer()
urwid.Frame.__init__(self, listbox, footer)

def keypress(self, size, key):
if key == 's':
content = self.get\_body().get\_text().get\_text()
with open(self.file\_name, 'w') as f:
f.write(content)
return key
if key == 'q':
raise urwid.ExitMainLoop()
if key == 'delete':
if self.get\_body().get\_focus() == len(self.get\_body().get\_text().get\_text()) - 1:
self.get\_body().get\_text().pop\_line()
else:
self.get\_body().get\_text().delete\_last\_character()
return key
if key == 'backspace':
if self.get\_body().get\_focus() == 0:
pass
else:
self.get\_body().get\_text().backspace()
return key
if key == 'enter':
self.get\_body().get\_text().newline()
return key
if key == 'up':
return urwid.cursor_up
if key == 'down':
return urwid.cursor\_down

def main(file\_name):
my\_text\_editor = MyTextEditor(file\_name)
top = urwid.Overlay(my\_text\_editor, urwid.SolidFill(u'\N{MEDIUM SHADE}'),
align='center', width=('relative', 80),
height=('relative', 80),
min_box_width=20, min_box_height=10)
urwid.MainLoop(top, unhandled\_input=my\_text\_editor.keypress).run()

if **name** == '**main**':
import sys
main(sys.argv[1])