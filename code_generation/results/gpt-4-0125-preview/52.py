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
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.create_new_explosion()

    def create_new_explosion(self):
        # Randomly generate points and colors for the explosion
        n = np.random.randint(100, 200)
        positions = np.random.rand(n, 2) * 2.0 - 1.0
        colors = np.random.rand(n, 3)
        self.program['a_position'] = gloo.VertexBuffer(positions)
        self.program['a_color'] = gloo.VertexBuffer(colors)
        self.explosion_start_time = self.timer.elapsed()

    def on_draw(self, event):
        gloo.clear('black')
        self.program.draw('points')

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_timer(self, event):
        if event.elapsed - self.explosion_start_time > 1:
            self.create_new_explosion()
        self.update()

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()
```