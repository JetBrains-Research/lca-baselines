 Here is a basic example of how you might structure your code. Note that this is a simplified version and does not include all the features you requested, but it should give you a good starting point.

```python
from vispy import app, gloo, shaders

class Firework(object):
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.life = 0.0
        self.size = 1.0

class Canvas(app.Canvas):
    def __init__(self, width=800, height=600):
        super().__init__(title='Fireworks', size=(width, height))
        self.fireworks = []
        self.timer = 0.0
        self.vbo = gloo.VertexBuffer(4, gloo.FLOAT, usage=gloo.STATIC)
        self.shader_program = shaders.compileFiles(
            ['firework_vertex.glsl', 'firework_fragment.glsl'],
            compiler='glslang',
            links=['firework'],
            flags=gloo.GL_FRAGMENT_PRECISION_HIGH
        )

    def on_key_press(self, event):
        if event.key == 'space':
            self.create_explosion()

    def create_explosion(self):
        pos = [self.width / 2, self.height / 2, 0]
        vel = [0, 0, 10]
        self.fireworks.append(Firework(pos, vel))

    def update(self, dt):
        self.timer += dt
        for firework in self.fireworks:
            firework.life += dt
            firework.size = firework.life / 1.0
            if firework.life > 1.0:
                self.fireworks.remove(firework)

        self.vbo.update(0, self.fireworks)

    def on_resize(self, event):
        self.width = event.new_size[0]
        self.height = event.new_size[1]

    def draw(self):
        with self.shader_program:
            self.vbo.bind()
            self.gl_points(4)
            self.clear(color=color.black)

            for firework in self.fireworks:
                self.set_uniform('u_time', self.timer)
                self.set_uniform('u_firework', firework)
                self.draw_arrays(gloo.POINTS, 0, 4)

class firework_vertex(shaders.GLSLProgram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_attribute('a_position')
        self.add_uniform('u_time', gloo.FLOAT)
        self.add_uniform('u_firework', gloo.STRUCT)

    def initialize(self, gl_context):
        self.bind()
        self.a_position = self.attribute_location('a_position')
        self.u_time = gl_context[0].get_uniform_location(self, 'u_time')
        self.u_firework = gl_context[0].get_uniform_location(self, 'u_firework')
        self.vertex_shader = self.create_shader(gl_context.GL_VERTEX_SHADER,
                                                 """
                                                 attribute vec3 a_position;
                                                 uniform float u_time;
                                                 uniform Firework;
                                                 void main() {
                                                     gl_Position = vec4(Firework.pos + Firework.vel * sin(u_time) * Firework.size, 0, 1);
                                                 }
                                                 """)
        self.fragment_shader = self.create_shader(gl_context.GL_FRAGMENT_SHADER,
                                                    """
                                                    void main() {
                                                        gl_FragColor = vec4(Firework.pos, Firework.size, Firework.size, 1.0);
                                                    }
                                                    """)
        self.link()

class Firework(object):
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        self.size = 1.0

if __name__ == '__main__':
    c = Canvas()
    c.show()
```

This code creates a basic fireworks simulation where each explosion is a point that moves in a sinusoidal pattern. The size of the point represents the age of the explosion. The `Firework` class represents a single explosion, and the `Canvas` class manages the simulation.

The vertex and fragment shaders are defined in the `firework_vertex` class. The vertex shader calculates the position of each point based on the time, the velocity, and the size of the explosion. The fragment shader sets the color of each point based on the position and size of the explosion.

The simulation is interactive and the window size is 800x600 pixels. Pressing the space bar creates a new explosion.

Please note that you will need to create the `firework_fragment.glsl` file to complete the shaders.

Also, this code does not include the optimization using Vertex Buffer Object (VBO) as requested. For that, you would need to update the `update` method to update the VBO with the new positions of the fireworks.