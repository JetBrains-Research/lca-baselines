```python
import numpy as np
from vispy import app, gloo

vertex_shader = """
attribute vec2 a_position;
attribute vec3 a_color;
varying vec3 v_color;
void main (void) {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_color = a_color;
}
"""

fragment_shader = """
varying vec3 v_color;
void main() {
    gl_FragColor = vec4(v_color, 1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, title='Fireworks Simulation', size=(800, 600))
        self.program = gloo.Program(vertex_shader, fragment_shader)
        self.vertices = np.zeros(100, [('a_position', np.float32, 2),
                                       ('a_color', np.float32, 3)])
        self.vertices_vbo = gloo.VertexBuffer(self.vertices)
        self.program.bind(self.vertices_vbo)
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.init_simulation()

    def init_simulation(self):
        self.create_new_explosion()

    def create_new_explosion(self):
        self.vertices['a_position'] = np.random.uniform(-1, 1, (100, 2))
        self.vertices['a_color'] = np.random.uniform(0, 1, (100, 3))
        self.vertices_vbo.set_data(self.vertices)

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        gloo.clear('black')
        self.program.draw('points')

    def on_timer(self, event):
        self.create_new_explosion()
        self.update()

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()
```