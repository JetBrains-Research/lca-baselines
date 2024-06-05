```python
from vispy import gloo, app
import numpy as np

vertex = """
attribute vec2 position;
void main (void)
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment = """
void main()
{
    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(800, 600), keys='interactive')
        self.program = gloo.Program(vertex, fragment)
        self.timer = app.Timer(1.0, connect=self.on_timer, start=True)
        self.initialize_simulation()

    def initialize_simulation(self):
        # Initialize simulation here
        pass

    def on_draw(self, event):
        gloo.clear(color='black')
        self.program.draw('points')

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_timer(self, event):
        self.create_explosion()

    def create_explosion(self):
        # Create a new explosion here
        pass

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()
```