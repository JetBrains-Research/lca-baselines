```python
import urwid

def show_menu(button, choices):
    body = [urwid.Text("Menu")]
    for c in choices:
        button = urwid.Button(c)
        urwid.connect_signal(button, 'click', item_chosen, c)
        body.append(urwid.AttrMap(button, None, focus_map='reversed'))
    main.original_widget = urwid.ListBox(urwid.SimpleFocusListWalker(body))

def item_chosen(button, choice):
    response = urwid.Text([u'You chose to ', choice, u'\n'])
    done = urwid.Button(u'Ok')
    urwid.connect_signal(done, 'click', exit_program)
    main.original_widget = urwid.Filler(urwid.Pile([response, urwid.AttrMap(done, None, focus_map='reversed')]))

def exit_program(button):
    raise urwid.ExitMainLoop()

palette = [
    ('reversed', 'standout', ''),
]

choices = ['Display Settings', 'Cursor Settings', 'LED Settings', 'About this Demo']
main = urwid.Padding(urwid.Filler(urwid.Text(u'Press any key\n')), left=2, right=2)
top = urwid.Overlay(main, urwid.SolidFill(u'\N{MEDIUM SHADE}'),
                    align='center', width=('relative', 60),
                    valign='middle', height=('relative', 60),
                    min_width=20, min_height=9)

urwid.MainLoop(top, palette, unhandled_input=show_menu, pop_ups=True).run()
```