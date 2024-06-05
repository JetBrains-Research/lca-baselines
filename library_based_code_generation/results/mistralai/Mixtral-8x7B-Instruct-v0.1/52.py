 ```python
import vispy.app as app
from vispy.geometry import create_box_outline
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Visual
from OpenGL.GL import *
import numpy as np


class Firework(Visual):
    def __init__(self, center, color, **kwargs):
        super(Firework, self).__init__(**kwargs)
        self.center = center
        self.color = color
        self.vertices = create_box_outline(1, 1, 1, 64).astype(np.float32)
        self.vertices = np.array(
            [(v[0] + 0.5, v[1] + 0.5, v[2] + 0.5, self.color[0], self.color[1], self.color[2], 1.0) for v in self.vertices],
            dtype=np.float32,
        )
        self.vbo = self.create_vbo(self.vertices)

    def set_position(self, position):
        self.vertices[:3] = position + 0.5
        self.vbo.set_data(self.vertices)

