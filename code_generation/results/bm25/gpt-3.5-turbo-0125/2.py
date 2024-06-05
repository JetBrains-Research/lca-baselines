import numpy
import argparse
import matplotlib
import scipy
import fealpy

parser = argparse.ArgumentParser()
parser.add_argument("--time_division", type=int, help="Time division")
parser.add_argument("--initial_spatial_division", type=int, help="Initial spatial division")
parser.add_argument("--num_spatial_iterations", type=int, help="Number of spatial iterations")
args = parser.parse_args()

mesh = fealpy.TriangleMesh()

pde = fealpy.ADI_2d()

ef_space = fealpy.FirstKindNedelecFiniteElementSpace2d(mesh)
mf_space = fealpy.ScaledMonomialSpace2d(mesh)

def get_phi_curl_matrix():
    pass

mass_matrix = fealpy.assembly_for_sspace_and_vspace_with_vector_basis(mesh, ef_space, ef_space)
curl_matrix = fealpy.assembly_for_sspace_and_vspace_with_vector_basis(mesh, ef_space, mf_space)

for i in range(args.num_spatial_iterations):
    rhs = compute_rhs_next_time_layer()
    handle_boundary_conditions_next_time_layer()
    compute_fields_next_time_layer()
    error_electric = calculate_error_electric()
    error_magnetic = calculate_error_magnetic()

    if i != args.num_spatial_iterations - 1:
        mesh.uniform_refine()

display_error_matrix()
plot_error_rates()