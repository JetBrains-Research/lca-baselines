import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import sparse
import fealpy

parser = argparse.ArgumentParser()
parser.add_argument("--time_division", type=int, default=100, help="Number of time divisions")
parser.add_argument("--initial_spatial_division", type=int, default=4, help="Initial spatial division")
parser.add_argument("--num_spatial_iterations", type=int, default=10, help="Number of spatial iterations")
args = parser.parse_args()

mesh = fealpy.TriangleMesh(unit='')

pde = fealpy.ADI_2d()
pde.setPDEParameters()

V = fealpy.FirstKindNedelecFiniteElementSpace2d(mesh)
Q = fealpy.ScaledMonomialSpace2d(mesh)

def get_phi_curl_matrix():
    pass

M = fealpy.FEMassMatrix(V)
C = get_phi_curl_matrix()

for i in range(args.num_spatial_iterations):
    rhs = pde.rightHandSide()
    pde.handleBoundaryCondition()
    E, H = pde.computeFields()
    error_E = pde.computeError(E)
    error_H = pde.computeError(H)

    if i != args.num_spatial_iterations - 1:
        mesh.uniformRefine()

fealpy.showerror(error_E, error_H)
fealpy.showrate(error_E, error_H)