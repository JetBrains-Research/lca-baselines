 ```python
import fealpy
import numpy as np
import time

class BoxDomain2D:
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly

    def create_mesh(self, h=0.1):
        return fealpy.mesh.TriangleMesh.box(self.Lx, self.Ly, h)

    def displacement(self, x, y):
        return np.zeros(2)

    def strain(self, x, y):
        u = self.displacement(x, y)
        return np.array([[u[0,:].mean(), u[1,:].mean()],
                         [u[1,:].mean(), u[0,:].mean()]]).T

    def stress(self, x, y):
        e = self.strain(x, y)
        return np.array([[e[0,:], e[1,:]],
                         [e[1,:], e[0,:]]])

    def source(self, x, y):
        return np.zeros(x.shape[1])

    def is_dirichlet_boundary(self, x, y):
        return np.logical_or(np.isclose(x, 0), np.isclose(x, self.Lx))

    def is_neumann_boundary(self, x, y):
        return np.isclose(y, 0)
