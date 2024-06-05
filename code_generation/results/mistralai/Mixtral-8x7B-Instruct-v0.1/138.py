 ```python
import fealpy.mesh as mesh
import fealpy.functionspace as space
import fealpy.function as fun
import numpy as np
import time
import matplotlib.pyplot as plt

class BoxDomain:
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly

    def create_mesh(self, nx=10, ny=10):
        self.mesh = mesh.BoxMesh(self.Lx, self.Ly, nx, ny)
