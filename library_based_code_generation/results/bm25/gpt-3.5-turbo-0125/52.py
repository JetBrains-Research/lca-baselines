```python
from vispy import app
from vispy.gloo import Program, VertexBuffer
from vispy.util.transforms import perspective, translate

import numpy as np

vertex = """
attribute vec2 a_position;
void main (void)
{
    gl_Position = vec4(a_position, 0.0, 1.0);
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
        self.program = Program(vertex, fragment)
        self.vertices = np.random.rand(100, 2).astype(np.float32)
        self.vbo = VertexBuffer(self.vertices)
        self.program['a_position'] = self.vbo
        self.show()

    def on_draw(self, event):
        self.program.draw('points')

    def on_resize(self, event):
        width, height = event.size
        self.program['u_projection'] = perspective(45.0, width / float(height), 2.0, 10.0)
        self.program['u_modelview'] = translate((0, 0, -5))

    def on_timer(self, event):
        self.create_explosion()

    def create_explosion(self):
        new_vertices = np.random.rand(100, 2).astype(np.float32)
        self.vbo.set_data(new_vertices)

if __name__ == '__main__':
    c = Canvas()
    app.run()
```