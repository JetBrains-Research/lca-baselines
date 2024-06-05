 ```python
import vispy.app as app
from vispy.geometry import create_cube
from vispy.scene import SceneCanvas
from vispy.visuals import Visual, VertexBuffer
import numpy as np

class Canvas(app.Canvas):
    def __init__(self, width=800, height=600, title="Fireworks Simulation"):
        super().__init__(width, height, title, render_method="gl2")
        self.timer = app.Timer(connect=self.on_timer, start=True)
        self.scene = SceneCanvas(keys='interactive')
        self.view = self.scene.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 90
        self.particles = []

        self.vertex_shader = """
        attribute vec3 vposition;
        varying vec3 fcolor;
        void main() {
            fcolor = vposition;
            gl_Position = vec4(vposition, 1.0);
        }
        """

        self.fragment_shader = """
        varying vec3 fcolor;
        void main() {
            gl_FragColor = vec4(fcolor, 1.0);
        }
        """

        self.program = self.view.program = self.view.scene.create_program(vertex_shader=self.vertex_shader,
                                                                           fragment_shader=self.fragment_shader)

    def on_timer(self, event):
        self.update()
        self.draw()

    def init_simulation(self):
        self.particles = []
        self.create_explosion()

    def create_explosion(self):
        pos = np.random.rand(100, 3) * 2 - 1
        vel = np.random.rand(100, 3) * 2
        col = np.random.rand(100, 3)
        self.particles.append(Particle(pos, vel, col))

    def draw_simulation(self, canvas):
        for particle in self.particles:
            particle.draw(canvas.program)

    def resize_simulation(self, event):
        glViewport(0, 0, event.width, event.height)

    def update_simulation(self, event):
        for particle in self.particles:
            particle.update()

    def main(self):
        self.init_simulation()
        app.run()

class Particle(Visual):
    def __init__(self, pos, vel, col):
        self.pos = pos
        self.vel = vel
        self.col = col
        self.age = 0
        self.lifespan = 1
        self.vertices = create_cube(pos, 0.05)
        self.vbo = VertexBuffer(self.vertices)
        super().__init__(vertexbuffer=self.vbo)

    def update(self):
        self.age += 1
        if self.age > self.lifespan:
            self.pos = np.random.rand(100, 3) * 2 - 1
            self.vel = np.random.rand(100, 3) * 2
            self.col = np.random.rand(100, 3)
            self.age = 0

        self.vertices[:, 0] = self.pos[:, 0] + self.vel[:, 0] * (self.lifespan - self.age) / self.lifespan
        self.vertices[:, 1] = self.pos[:, 1] + self.vel[:, 1] * (self.lifespan - self.age) / self.lifespan
        self.vertices[:, 2] = self.pos[:, 2] + self.vel[:, 2] * (self.lifespan - self.age) / self.lifespan

    def draw(self, program):
        program['vposition'] = self.vertices
        glEnableVertexAttribArray(program['vposition'].location)
        glVertexAttribPointer(program['vposition'].location, 3, GL_FLOAT, GL_FALSE, 0, self.vbo)
        glDrawArrays(GL_POINTS, 0, len(self.vertices))

if __name__ == '__main__':
    c = Canvas()
    c.main()
```
Please note that this code is a basic implementation of the fireworks simulation using vispy library and it may not be fully optimized. The explosion is simulated by changing the position of the particles in each frame, and the color and lifetime of each particle are randomly generated. The code also includes a simple update method that checks the age of the particle and resets its position, velocity, and color if its age exceeds its lifespan. The draw method of the particle class uses the vertex buffer object to efficiently render the particles in the scene.