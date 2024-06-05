from urwid import Frame, Filler, ListBox, Pile, Text, connect\_keypressers, Quit, AttrMap, editline, Overlay, PopUpLoop
from urwid.raw\_display import RawDisplay
import urwid.curses_display
import os

class LazyLoader(ListBox):
def __init__(self, func):
self.func = func
super(LazyLoader, self).__init__([self.body])

@property
def body(self):
return self._body

@body.setter
def body(self, value):
self._body = value
self.set\_size(None, len(value))

def keypress(self, size, key):
if key == "down":
new\_index = self.get\_focus() + 1
if new\_index < len(self.body):
self.set\_focus(new\_index)
return key
elif key == "up":
new\_index = self.get\_focus() - 1
if new\_index >= 0:
self.set\_focus(new\_index)
return key
elif key == "page down":
new\_index = min(len(self.body), self.get\_focus() + (size[0] - 1))
self.set\_focus(new\_index)
return key
elif key == "page up":
new\_index = max(0, self.get\_focus() - (size[0] - 1))
self.set\_focus(new\_index)
return key
elif key == "home":
self.set\_focus(0)
return key
elif key == "end":
self.set\_focus(len(self.body) - 1)
return key
return key

class MyTextEditor(object):
def __init__(self, filename):
self.filename = filename
self.edited = False
self.text = ""
self.maxcol = 80
self.cursor\_pos = (0, 0)
self.original\_text = ""

if os.path.exists(filename):
with open(filename, "r") as f:
self.original\_text = f.read()
self.text = self.original\_text

self.listbox = LazyLoader(self.build\_listbox)
self.footer = Text("Type 'q' to quit, 's' to save.", align="center")

self.pile = Pile([self.listbox, self.footer])
self.filler = Filler(self.pile, "top")
self.main\_widget = Frame(self.filler, footer=self.footer)

self.edit\_text = self.main\_widget[0].original\_widget.body

self.edit\_text.set\_edit\_pos(self.cursor\_pos)
self.edit\_text.edit\_pos = self.cursor\_pos
self.edit\_text.set\_callback(self.on\_keypress)

self.loop = urwid.MainLoop(self.main\_widget, palette=[("reverse", "standout", "")])

def build\_listbox(self):
text\_list = self.text.split("\n")
return [Text(line) for line in text\_list]

def on\_keypress(self, size, key):
if key == "q":
raise Quit()
elif key == "s":
self.save\_to\_file()
elif key == "enter":
self.new\_line()
elif key == "backspace":
if self.cursor\_pos[1] == 0 and self.cursor\_pos[0] > 0:
self.text = self.text[: self.cursor\_pos[0] - 1] + "\n" + self.text[self.cursor\_pos[0]:]
self.cursor\_pos = (self.cursor\_pos[0] - 1, self.maxcol)
self.edit\_text.set\_edit\_pos(self.cursor\_pos)
self.edit\_text.edit\_pos = self.cursor\_pos
elif key == "delete":
if self.cursor\_pos[1] < self.maxcol:
self.text = (
self.text[: self.cursor\_pos[0]]
+ self.text[self.cursor\_pos[0] + 1 :]
)
self.edit\_text.set\_edit\_text(self.text)
self.edit\_text.set\_edit\_pos((self.cursor\_pos[0], self.cursor\_pos[1] - 1))
self.edit\_text.edit\_pos = (self.cursor\_pos[0], self.cursor\_pos[1] - 1)
elif key == "left":
if self.cursor\_pos[1] > 0:
self.cursor\_pos = (self.cursor\_pos[0], self.cursor\_pos[1] - 1)
self.edit\_text.set\_edit\_pos(self.cursor\_pos)
elif self.cursor\_pos[0] > 0:
self.cursor\_pos = (self.cursor\_pos[0] - 1, self.maxcol)
self.edit\_text.set\_edit\_pos(self.cursor\_pos)
elif key == "right":
if self.cursor\_pos[1] < self.maxcol:
self.cursor\_pos = (self.cursor\_pos[0], self.cursor\_pos[1] + 1)
self.edit\_text.set\_edit\_pos(self.cursor\_pos)
elif self.cursor\_pos[0] < len(self.text.split("\n")):
self.cursor\_pos = (self.cursor\_pos[0] + 1, 0)
self.edit\_text.set\_edit\_pos(self.cursor\_pos)
elif key == "up":
if self.cursor\_pos[0] > 0:
self.cursor\_pos = (self.cursor\_pos[0] - 1, min(self.cursor\_pos[1], len(self.text.split("\n")[-1])))
self.edit\_text.set\_edit\_pos(self.cursor\_pos)
elif key == "down":
if self.cursor\_pos[0] < len(self.text.split("\n")):
self.cursor\_pos = (self.cursor\_pos[0] + 1, min(self.cursor\_pos[1], len(self.text.split("\n")[-1])))
self.edit\_text.set\_edit\_pos(self.cursor\_pos)
else:
self.edit\_text.set\_edit\_pos(self.cursor\_pos)

def new\_line(self):
if self.cursor\_pos[1] == self.maxcol:
self.text = self.text[: self.cursor\_pos[0]] + "\n" + self.text[self.cursor\_pos[0]:]
self.cursor\_pos = (self.cursor\_pos[0] + 1, 0)
self.edit\_text.set\_edit\_pos(self.cursor\_pos)
else:
self.text = self.text[: self.cursor\_pos[0]] + "\n" + self.text[self.cursor\_pos[0]:]
self.text = self.text[: self.cursor\_pos[0] + 1] + " " * (self.maxcol - self.cursor\_pos[1]) + self.text[self.cursor\_pos[0] + 1:]
self.cursor\_pos = (self.cursor\_pos[0] + 1, self.maxcol)
self.edit\_text.set\_edit\_pos(self.cursor\_pos)

def save\_to\_file(self):
with open(self.filename, "w") as f:
f.write(self.text)
self.edited = False

def run(self):
self.loop.run()

if **name** == "**main**":
filename = "example.txt"
MyTextEditor(filename).run()
\*\* Note: This code uses the urwid library, which must be installed before running. \*\*