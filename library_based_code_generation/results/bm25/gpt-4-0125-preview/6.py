```python
import urwid

def menu_button(caption, callback):
    button = urwid.Button(caption)
    urwid.connect_signal(button, 'click', callback)
    return urwid.AttrMap(button, None, focus_map='reversed')

def sub_menu(title, choices):
    body = [urwid.Text(title), urwid.Divider()]
    for c in choices:
        button = menu_button(c[0], c[1])
        body.append(urwid.AttrMap(button, None, focus_map='reversed'))
    return urwid.ListBox(urwid.SimpleFocusListWalker(body))

def main_menu():
    return sub_menu('Main Menu', [
        ('Display Settings', display_settings),
        ('Cursor Settings', cursor_settings),
        ('LED Settings', led_settings),
        ('About this Demo', about_demo),
    ])

def display_settings(button):
    response = urwid.Text([u'Adjust Brightness and Contrast here.\n'])
    done = menu_button(u'Back', exit_program)
    main.original_widget = urwid.Filler(urwid.Pile([response, done]))

def cursor_settings(button):
    response = urwid.Text([u'Choose Cursor Styles here.\n'])
    done = menu_button(u'Back', exit_program)
    main.original_widget = urwid.Filler(urwid.Pile([response, done]))

def led_settings(button):
    response = urwid.Text([u'Adjust LED Colors here.\n'])
    done = menu_button(u'Back', exit_program)
    main.original_widget = urwid.Filler(urwid.Pile([response, done]))

def about_demo(button):
    response = urwid.Text([u'This is a demo of a user interface for a crystalfontz 635 LCD display.\n'])
    done = menu_button(u'Back', exit_program)
    main.original_widget = urwid.Filler(urwid.Pile([response, done]))

def exit_program(button):
    raise urwid.ExitMainLoop()

main = urwid.Padding(main_menu(), left=2, right=2)
top = urwid.Overlay(main, urwid.SolidFill(u'\N{MEDIUM SHADE}'),
                    align='center', width=('relative', 60),
                    valign='middle', height=('relative', 60),
                    min_width=20, min_height=9)
urwid.MainLoop(top, palette=[('reversed', 'standout', '')]).run()
```