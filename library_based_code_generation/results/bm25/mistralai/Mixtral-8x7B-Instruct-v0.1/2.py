 ```python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg
from fealpy.mesh import MeshFactory as MF
from fealpy.pde.advection import ADI_2d
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d, ScaledMonomialSpace2d
from fealpy.timeintegrator import TimeIntegrator
from fealpy.tool import test_assembly_cell_righthand_side_and_matrix, test_assembly_cell_righthand_side_and_dof_matrix, test_dirichlet_and_neumann_bc_on_interval_mesh, test_dirichlette_and_neumann_bc_on_tetrahedron_mesh, get_next_time, get_all_gauss_point_and_weight, BoundaryLayerField

parser = argparse.ArgumentParser()
parser.add_argument('--ndiv', type=int, default=3, help='time division')
parser.add_argument('--sdiv', type=int, default=4, help='initial spatial division')
parser.add_argument('--nstep', type=int, default=3, help='number of spatial iterations')
args = parser.parse_args()

mesh_factory = MF()
mesh = mesh_factory.create_mesh_2d(np.array([0, 0, 1, 1]), 'left', 'bottom', args.sdiv, 1)

pde = ADI_2d()
pde.set_domain(mesh)

electric_space = FirstKindNedelecFiniteElementSpace2d(mesh, p=1, rc=None, rd=None)
magnetic_space = ScaledMonomialSpace2d(mesh, p=1, rc=None, rd=None)

phi_curl_matrix = get_phi_curl_matrix(electric_space)

mass_matrix_e = test_assembly_cell_righthand_side_and_matrix(mesh, electric_space, pde.mass_matrix_formula)
mass_matrix_m = test_assembly_cell_righthand_side_and_matrix(mesh, magnetic_space, pde.mass_matrix_formula)
curl_matrix_e = test_assembly_cell_righthand_side_and_matrix(mesh, electric_space, pde.curl_matrix_formula)

def get_phi_curl_matrix(space):
    phi_curl_matrix = np.zeros((space.nunit_dof, space.nunit_dof))
    for i in range(space.nunit_cell):
        cell = space.unit_cells[i]
        phi_curl_matrix[cell.dofs, cell.dofs] = space.get_phi_curl_matrix(cell)
    return phi_curl_matrix

def get_next_time_layer(time_mesh, electric_field, magnetic_field, mass_matrix_e, curl_matrix_e, mass_matrix_m, phi_curl_matrix):
    nt = time_mesh.nt
    ne = electric_field.shape[0]
    nm = magnetic_field.shape[0]

    b1 = np.zeros(ne)
    b2 = np.zeros(ne)
    b3 = np.zeros(nm)

    for i in range(nt - 1):
        time_level = time_mesh.time_levels[i]
        time_next = time_mesh.time_levels[i + 1]

        electric_field_next = np.zeros(ne)
        magnetic_field_next = np.zeros(nm)

        for j in range(ne):
            dof = electric_field.loc[j]
            b1[j] += time_level.dt * (mass_matrix_e[dof, dof] * electric_field[dof] + curl_matrix_e[dof, dof] * magnetic_field[dof])
            b2[j] += time_level.dt * (phi_curl_matrix[dof, dof] * magnetic_field[dof])

        for j in range(nm):
            dof = magnetic_field.loc[j]
            b3[j] += time_level.dt * (-phi_curl_matrix[dof, dof] * electric_field[dof])

        electric_field_next[:] = linalg.spsolve(mass_matrix_e, b1)
        magnetic_field_next[:] = linalg.spsolve(mass_matrix_m, b3)

        electric_field_next[time_next.boundary_marker] = test_dirichlet_and_neumann_bc_on_interval_mesh(time_next, electric_field_next, time_next.boundary_marker, pde.dirichlet_formula, pde.neumann_formula)
        magnetic_field_next[time_next.boundary_marker] = test_dirichlette_and_neumann_bc_on_tetrahedron_mesh(time_next, magnetic_field_next, time_next.boundary_marker, pde.dirichlet_formula, pde.neumann_formula)

        electric_field = electric_field_next
        magnetic_field = magnetic_field_next

    return electric_field, magnetic_field

def get_error(exact_electric_field, exact_magnetic_field, electric_field, magnetic_field):
    error_e = np.linalg.norm(exact_electric_field - electric_field)
    error_m = np.linalg.norm(exact_magnetic_field - magnetic_field)
    return error_e, error_m

time_mesh = TimeIntegrator(args.ndiv)

exact_electric_field = BoundaryLayerField(mesh, 'zero')
exact_magnetic_field = BoundaryLayerField(mesh, 'zero')

electric_field = np.zeros(electric_space.nunit_dof)
magnetic_field = np.zeros(magnetic_space.nunit_dof)

for iter in range(args.nstep):
    time_mesh.uniform_refine()
    mesh.uniform_refine()

    electric_field, magnetic_field = get_next_time_layer(time_mesh, electric_field, magnetic_field, mass_matrix_e, curl_matrix_e, mass_matrix_m, phi_curl_matrix)

    error_e, error_m = get_error(exact_electric_field.data, exact_magnetic_field.data, electric_field, magnetic_field)
    print('error_e:', error_e)
    print('error_m:', error_m)

error_e_history = np.zeros(args.nstep + 1)
error_m_history = np.zeros(args.nstep + 1)
error_e_history[0] = error_e
error_m_history[0] = error_m

for iter in range(1, args.nstep + 1):
    error_e_history[iter] = error_e_history[iter - 1] / 2**(2 * iter)
    error_m_history[iter] = error_m_history[iter - 1] / 2**(2 * iter)

plt.loglog(time_mesh.time_nodes, error_e_history, label='error_e')
plt.loglog(time_mesh.time_nodes, error_m_history, label='error_m')
plt.legend()
plt.show()
```