from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC, NeumannBC
from fealpy.linear_elasticity import LinearElasticity
from fealpy.solver import solve

class BoxDomain:
    def __init__(self, box):
        self.box = box

    def init_mesh(self, n):
        self.mesh = MeshFactory.boxmesh2d(self.box, n)

    def displacement(self, p):
        pass

    def strain(self, p):
        pass

    def stress(self, p):
        pass

    def source(self, p):
        pass

    def is_on_Dirichlet_boundary(self, p):
        pass

    def is_on_Neumann_boundary(self, p):
        pass

    def is_on_fracture_boundary(self, p):
        pass

class IterationCounter:
    def __init__(self):
        self.iter = 0

class FastSolver:
    def __init__(self):
        pass

    def preconditioning(self):
        pass

    def solve_system(self):
        pass

box = [0, 1, 0, 1]
n = 4
domain = BoxDomain(box)
domain.init_mesh(n)

space = LagrangeFiniteElementSpace(domain.mesh, p=1)
dirichlet = DirichletBC(space, domain.is_on_Dirichlet_boundary)
neumann = NeumannBC(space, domain.is_on_Neumann_boundary)

def solution(p):
    pass

K = LinearElasticity.stiff_matrix(space)
C = LinearElasticity.elasticity_matrix(space)
f = LinearElasticity.source_vector(space, domain.source)

A, b = LinearElasticity.assemble_matrix_rhs(K, C, f)

A, b = dirichlet.apply(A, b)
A, b = neumann.apply(A, b)

print(A.shape)

solver = FastSolver()
start_time = time.time()
solver.solve_system(A, b)
end_time = time.time()
print("Time taken to solve the system:", end_time - start_time)

domain.mesh.add_plot(1)
domain.mesh.show()