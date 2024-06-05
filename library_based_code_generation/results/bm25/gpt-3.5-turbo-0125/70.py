import numpy as np
import scipy as sp
import porepy as pp

def add_data(grid_bucket, domain, permeability_factor):
    # Add parameters to the grid bucket
    pass

def write_network(file_name):
    # Write predefined network string to file
    pass

def main(permeability_factor, description, coarsen_grid_bucket, export_results):
    # Define mesh parameters, domain, and file name
    pass

def test_vem_blocking():
    main(small_permeability_factor, "blocking", True, False)

def test_vem_permeable():
    main(large_permeability_factor, "permeable", False, True)