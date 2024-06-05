import numpy as np
from scipy import sparse
import porepy as pp

def add_data(grid_bucket, domain, permeability_factor):
    # Add parameters to grid bucket
    pass

def write_network(file_name):
    # Write predefined network string to file
    pass

def main(permeability_factor, description, coarsen_grid_bucket, export_results):
    # Define mesh parameters, domain, and file name
    pass

def test_vem_blocking():
    main(0.1, "blocking", True, False)

def test_vem_permeable():
    main(10.0, "permeable", False, True)