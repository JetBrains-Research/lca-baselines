  import sys
import os
import numpy as np
from sirt import SIRF

def osmaposl_reconstruction(image, objective_function, prior, filter, num_subsets, num_sub_iterations, non_interactive=False):
    # Create acquisition model and data
    acquisition_model = SIRF.AcquisitionModel()
    acquisition_data = SIRF.AcquisitionData()
    acquisition_model.set_objective_function(objective_function)
    acquisition_data.set_objective_function(objective_function)
    acquisition_model.set_maximum_number_of_sigmas(3)
    acquisition_data.set_maximum_number_of_sigmas(3)
    acquisition_model.name_and_parameters()
    acquisition_data.name_and_parameters()
    acquisition_model.test_sti_objective_function()
    acquisition_data.test_sti_objective_function()
    acquisition_model.number()
    acquisition_data.number()
    acquisition_model.normalise_zero_and_one()
    acquisition_data.normalise_zero_and_one()
    acquisition_model.value_of()
    acquisition_data.value_of()
    acquisition_model.label_and_name()
    acquisition_data.label_and_name()
    acquisition_model.field_of_view()
    acquisition_data.field_of_view()
    acquisition_model.get_backprojection_of_acquisition_ratio()
    acquisition_data.get_backprojection_of_acquisition_ratio()

    # Create image data
    image_data = SIRF.ImageData()
    image_data.set_image(image)

    # Create filter
    filter = SIRF.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()

    # Create prior
    prior = SIRF.ImageData()
    prior.set_image(np.zeros(image.shape))

    # Perform reconstruction
    for i in range(num_sub_iterations):
        for j in range(num_subsets):
            subset = acquisition_data.subset(j, num_subsets)
            subset.set_objective_function(objective_function)
            subset.set_maximum_number_of_sigmas(3)
            subset.name_and_parameters()
            subset.test_sti_objective_function()
            subset.number()
            subset.normalise_zero_and_one()
            subset.value_of()
            subset.label_and_name()
            subset.field_of_view()
            subset.get_backprojection_of_acquisition_ratio()
            image_data.set_image(subset.reconstruct(image_data, filter, prior))

    # Display reconstructed image if non-interactive option is not set
    if not non_interactive:
        print("Reconstructed image:")
        print(image_data.image())

if __name__ == "__main__":
    # Parse command-line options
    raw_data_file = sys.argv[1]
    data_path = sys.argv[2]
    num_subsets = int(sys.argv[3])
    num_sub_iterations = int(sys.argv[4])
    reconstruction_engine = sys.argv[5]
    non_interactive = sys.argv[6] == "True"

    # Load raw data
    data = np.load(raw_data_file)

    # Create acquisition model and data
    acquisition_model = SIRF.AcquisitionModel()
    acquisition_data = SIRF.AcquisitionData()
    acquisition_model.set_objective_function(SIRF.ObjectiveFunction())
    acquisition_data.set_objective_function(SIRF.ObjectiveFunction())
    acquisition_model.set_maximum_number_of_sigmas(3)
    acquisition_data.set_maximum_number_of_sigmas(3)
    acquisition_model.name_and_parameters()
    acquisition_data.name_and_parameters()
    acquisition_model.test_sti_objective_function()
    acquisition_data.test_sti_objective_function()
    acquisition_model.number()
    acquisition_data.number()
    acquisition_model.normalise_zero_and_one()
    acquisition_data.normalise_zero_and_one()
    acquisition_model.value_of()
    acquisition_data.value_of()
    acquisition_model.label_and_name()
    acquisition_data.label_and_name()
    acquisition_model.field_of_view()
    acquisition_data.field_of_view()
    acquisition_model.get_backprojection_of_acquisition_ratio()
    acquisition_data.get_backprojection_of_acquisition_ratio()

    # Create image data
    image_data = SIRF.ImageData()
    image_data.set_image(data)

    # Create filter
    filter = SIRF.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()

    # Create prior
    prior = SIRF.ImageData()
    prior.set_image(np.zeros(data.shape))

    # Perform reconstruction
    try:
        osmaposl_reconstruction(image_data, acquisition_model, prior, filter, num_subsets, num_sub_iterations, non_interactive)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)