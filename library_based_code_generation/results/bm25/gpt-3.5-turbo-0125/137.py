import numpy as np
import scipy
import os
import sys
from porepy import *

def add_data_darcy(gb, tol):
    # Add parameters related to Darcy's law
    pass

def add_data_advection(gb, tol):
    # Add parameters related to advection
    pass

sys.path.append('path_to_module')
import soultz_grid

# Set up parameters for creating a grid
# Create grid using soultz_grid module
# Compute geometry, coarsen grid, assign node ordering

# Solve Darcy problem using DualVEMMixDim solver
# Add parameters to grid bucket
# Compute matrix and right-hand side vector
# Solve system of equations
# Split solution, extract discharge and pressure
# Project discharge, compute total flow rate

# Set up parameters for transport problem
# Define solvers for advection and mass matrix
# Add parameters to grid bucket
# Compute matrices and right-hand side vectors
# Perform LU factorization
# Initialize solution vector
# Time-stepping loop to update solution and export at certain time steps

# Export solution in PVD format
# Save production data to text file