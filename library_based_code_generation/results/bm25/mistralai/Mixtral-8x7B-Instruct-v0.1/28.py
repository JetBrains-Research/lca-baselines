 import argparse
import sys
from sirf.Filtering import Reconstruction
from sirf.ObjectiveFunction import PoissonLogLikelihoodWithLinearModelForMeanAndProjData
from sirf.Utilities import normalise_zero_and_one
from sirf.PhysicsEngines import get_backprojection_of_acquisition_ratio
from sirf.Image import ImageData
from sirf.DataHandling import DataHandleVector

def osmaposl(image, objective_function, prior, filter, n_subsets, n_sub_iterations):
"""
User-defined OSMAPOSL reconstruction algorithm.
"""
for subset in range(n_subsets):
for sub_iteration in range(n_sub_iterations):
image = filter.one_step_late_map_estimate(image, objective_function, prior)
return image

if __name__ == "__main__":
parser = argparse.ArgumentParser(description='OSMAPOSL reconstruction.')
parser.add_argument('raw_data_file', help='Path to the raw data file.')
parser.add_argument('path_to_data_files', help='Path to the data files.')
parser.add_argument('--num_subsets', type=int, default=5, help='Number of subsets (default: 5).')
parser.add_argument('--num_sub_iterations', type=int, default=10, help='Number of sub-iterations (default: 10).')
parser.add_argument('--recon_engine', type=str, default='OSEM', help='Reconstruction engine (default: OSEM).')
parser.add_argument('--no_plots', action='store_true', help='Disable plots.')
args = parser.parse_args()

try:
# Create acquisition model
acq_model = get_backprojection_of_acquisition_ratio(args.raw_data_file, args.path_to_data_files)

# Create acquisition data
acq_data = AcquisitionData(acq_model)

# Create filter
filter_obj = Reconstruction(acq_model, args.recon_engine)

# Create initial image estimate
initial_image = ImageData(field_of_view=acq_model.get_field_of_view())
initial_image.create_image_data(number=acq_model.get_number_of_samples())

# Create prior
prior_obj = prior.Prior(initial_image)

# Create objective function
objective_function = PoissonLogLikelihoodWithLinearModelForMeanAndProjData(acq_data, prior_obj)

# Set maximum number of sigmas
objective_function.set_maximum_number_of_sigmas(3.0)

# Normalize zero and one
normalise_zero_and_one(initial_image)

# Call the reconstruction function
reconstructed_image = osmaposl(initial_image, objective_function, prior_obj, filter_obj, args.num_subsets, args.num_sub_iterations)

# Display the reconstructed image if the non-interactive option is not set
if not args.no_plots:
reconstructed_image.show()

except Exception as e:
print("Error:", e)
sys.exit(1)