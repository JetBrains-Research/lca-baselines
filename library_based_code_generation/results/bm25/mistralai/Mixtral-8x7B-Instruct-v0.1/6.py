 import urwid

class CursoredEdit(urwid.Edit, urwid.WidgetWrap):
pass

class CheckBox(urwid.WidgetWrap):
def __init__(self, label, state=False):
self.state = state
self.label = label

self.widget = urwid.Text(self.get\_text())

def get\_text(self):
return urwid.AttrMap(self.widget, None if self.state else "disabled")

def set\_state(self, state):
self.state = state
self.widget = urwid.Text(self.get\_text())

def keypress(self, size, key):
if key == "space":
self.state = not self.state
self.widget = urwid.Text(self.get\_text())
return key

class RadioButton(urwid.WidgetWrap):
def __init__(self, label, group, value):
self.label = label
self.group = group
self.value = value

self.widget = urwid.Text(self.get\_text())

def get\_text(self):
return urwid.AttrMap(self.widget, None if self.value == self.group else "disabled")

def keypress(self, size, key):
if key == "space":
self.group.value = self.value
self.widget = urwid.Text(self.get\_text())
return key

class ProgressBarWithCustomChars(urwid.ProgressBar):
def paint\_bar(self, size, start):
progress, max\_progress = self.get\_progress()
if progress > max\_progress:
progress = max\_progress

left\_edge = self.opt[0]
right\_edge = self.opt[1]

bar\_width = right\_edge - left\_edge - 2
bar\_length = int(bar\_width \* progress / max\_progress)

bar\_chars = "|" * bar\_length + " " * (bar\_width - bar\_length)

return left\_edge + bar\_chars + right\_edge

class Slider(urwid.WidgetWrap):
def __init__(self, min\_value, max\_value, value, caption, on\_change):
self.min\_value = min\_value
self.max\_value = max\_value
self.value = value
self.caption = caption
self.on\_change = on\_change

self.widget = urwid.AttrMap(urwid.Columns([
("weight", 1, urwid.Text(caption)),
("weight", 10, urwid.Edit(self.value, callback=self.on\_change))
], dividechars=1), None)

class DisplaySettings(urwid.WidgetWrap):
def __init__(self, on\_change):
self.on\_change = on\_change

self.brightness\_edit = CursoredEdit("", callback=self.on\_change)
self.contrast\_edit = CursoredEdit("", callback=self.on\_change)

self.widget = urwid.AttrMap(urwid.Columns([
("weight", 1, urwid.Text("Brightness")),
("weight", 10, self.brightness\_edit),
("weight", 1, urwid.Text("Contrast")),
("weight", 10, self.contrast\_edit),
]), None)

class CursorSettings(urwid.WidgetWrap):
def __init__(self, on\_change):
self.on\_change = on\_change

self.widget = urwid.AttrMap(urwid.RadioButton("Style 1", self, 1), None)

class LedSettings(urwid.WidgetWrap):
def __init__(self, on\_change):
self.on\_change = on\_change

self.red\_edit = CursoredEdit("", callback=self.on\_change)
self.green\_edit = CursoredEdit("", callback=self.on\_change)
self.blue\_edit = CursoredEdit("", callback=self.on\_change)

self.widget = urwid.AttrMap(urwid.Columns([
("weight", 1, urwid.Text("Red")),
("weight", 10, self.red\_edit),
("weight", 1, urwid.Text("Green")),
("weight", 10, self.green\_edit),
("weight", 1, urwid.Text("Blue")),
("weight", 10, self.blue\_edit),
]), None)

class AboutThisDemo(urwid.WidgetWrap):
def __init__(self):
self.widget = urwid.Text("About this Demo\n\nThis is a demo of a user interface for a Crystalfontz 635 LCD display using the urwid library in Python.")

class CustomCharacters(urwid.WidgetWrap):
def __init__(self):
self.check\_box = CheckBox("Check box")
self.radio\_button1 = RadioButton("Radio button 1", self, 1)
self.radio\_button2 = RadioButton("Radio button 2", self, 2)
self.progress\_bar = ProgressBarWithCustomChars(40, 100, 60)
self.slider = Slider(0, 100, 50, "Slider", self.on\_slider\_change)

def on\_slider\_change(self, new\_value):
print("Slider changed to", new\_value)

class MainMenu(urwid.WidgetWrap):
def __init__(self, on\_change):
self.on\_change = on\_change

self.display\_settings\_button = MenuButton("Display settings", self.display\_settings\_clicked)
self.cursor\_settings\_button = MenuButton("Cursor settings", self.cursor\_settings\_clicked)
self.led\_settings\_button = MenuButton("LED settings", self.led\_settings\_clicked)
self.about\_this\_demo\_button = MenuButton("About this Demo", self.about\_this\_demo\_clicked)
self.custom\_characters\_button = MenuButton("Custom characters", self.custom\_characters\_clicked)

self.widget = urwid.AttrMap(urwid.Columns([
("weight", 1, self.display\_settings\_button),
("weight", 1, self.cursor\_settings\_button),
("weight", 1, self.led\_settings\_button),
("weight", 1, self.about\_this\_demo\_button),
("weight", 1, self.custom\_characters\_button),
]), None)

def display\_settings\_clicked(self, button):
self.on\_change("display\_settings")

def cursor\_settings\_clicked(self, button):
self.on\_change("cursor\_settings")

def led\_settings\_clicked(self, button):
self.on\_change("led\_settings")

def about\_this\_demo\_clicked(self, button):
self.on\_change("about\_this\_demo")

def custom\_characters\_clicked(self, button):
self.on\_change("custom\_characters")

if **name** == "**main**":
urwid.MainLoop(MainMenu(lambda x: None), unhandled\_input=lambda x: None).run()