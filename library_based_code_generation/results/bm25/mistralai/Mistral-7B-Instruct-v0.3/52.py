 Here is a basic implementation of the requested fireworks simulation using vispy and GLSL shaders. Please note that this is a simplified version and may require further optimization and adjustments for specific requirements.

```python
import vispy.app
from vispy.scene import graphics, visuals
from vispy.glyph import GLSLProgram

class Firework(visuals.GLVisual):
    def __init__(self, pos, life=1.0, size=10.0, color=(1, 1, 1, 1)):
        super().__init__()
        self.pos = pos
        self.life = life
        self.size = size
        self.color = color
        self.vbo = graphics.VertexBuffer(format=graphics.vertex_format_3d, interactive=False)
        self.vao = self.vbo.vao
        self.program = GLSLProgram(vertex=vertex_shader, fragment=fragment_shader)

    def update(self, event):
        self.life -= event.dt
        if self.life <= 0:
            self.delete()
        else:
            self.size *= 1.05
            self.life *= 0.99
            self.pos[1] += self.size * 0.05
            self.vbo.set_data(self.pos)

    def draw(self, camera):
        self.program.bind()
        self.program['u_time'] = event.time
        self.program['u_life'] = self.life
        self.program['u_size'] = self.size
        self.program['u_color'] = self.color
        self.vao.draw('points')
        self.program.unbind()

class Canvas(vispy.app.Canvas):
    def __init__(self, width=800, height=600):
        super().__init__(keys='interactive', show=True, width=width, height=height)
        self.fireworks = []
        self.timer = self.scene.timer
        self.timer.connect(self.on_timer)

    def on_resize(self, event):
        self.scene.active_camera.aspect = self.width / self.height

    def on_timer(self, event):
        self.fireworks.append(Firework([self.width / 2, self.height, 0], life=1.0))

    def create_new_explosion(self):
        pass  # Implement a method to create a new explosion with unique properties

    def run(self):
        self.show()
        self.app.run()

vertex_shader = """
#version 330

uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;

uniform float u_time;
uniform float u_life;
uniform float u_size;
uniform vec4 u_color;

in vec3 v_position;

void main() {
    gl_Position = u_proj * u_view * u_model * vec4(v_position, 1.0);
    gl_PointSize = u_size;
    gl_FragColor = u_color;
}
"""

fragment_shader = """
#version 330

out vec4 frag_color;

uniform float u_time;
uniform float u_life;
uniform float u_size;
uniform vec4 u_color;

void main() {
    float t = u_time * 0.001;
    float life_progress = clamp((u_time - u_life) / 1.0, 0.0, 1.0);
    float color_progress = clamp(u_life / 1.0, 0.0, 1.0);
    vec3 color = mix(vec3(1.0, 0.0, 0.0), u_color.rgb, color_progress);
    vec3 new_color = mix(color, vec3(0.0, 0.0, 0.0), smoothstep(0.5, 0.7, life_progress));
    frag_color = vec4(new_color, u_color.a);
}
"""

if __name__ == "__main__":
    app = Canvas()
    app.run()
```

This code creates a Canvas with a fireworks simulation. The fireworks are represented by the Firework class, which uses a Vertex Buffer Object (VBO) for optimization. The vertex and fragment shaders are defined in GLSL. The simulation creates a new explosion every time the timer ticks, and you can implement the `create_new_explosion` method to create unique explosions. The window size is set to 800x600 pixels.