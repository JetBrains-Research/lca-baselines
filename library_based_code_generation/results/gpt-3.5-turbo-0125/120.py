import sys
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.pde.poisson_2d import CosCosData

maxit = int(sys.argv[1])
theta = float(sys.argv[2])
k = int(sys.argv[3])

mesh = HalfEdgeMesh()
mesh = mesh.init_mesh()
space = LagrangeFiniteElementSpace(mesh, p=1)

pde = CosCosData()
uh = space.function()
uh[:] = pde.solution(mesh.node)

for i in range(maxit):
    uh, error = pde.solve(mesh, space, uh, theta, k)
    residuals = pde.residuals(mesh, space, uh)
    high_order_terms = pde.high_order_terms(mesh, space, uh)
    
    np.savetxt('results_iteration_{}.txt'.format(i), np.column_stack((uh, error)), fmt='%.6e')
    
    plt.figure()
    mesh.add_plot(plt)
    plt.savefig('mesh_iteration_{}.png'.format(i))
    
    if i != maxit-1:
        mesh.uniform_refine()
        
np.savetxt('final_error_data.txt', np.column_stack((uh, error)), fmt='%.6e')

plt.figure()
# plot multi-rate
plt.show()