  import os
import sys
import numpy as np
import sirtf as sirtf

def osmaposl_reconstruction(image, objective_function, prior, filter, num_subsets, num_sub_iterations, non_interactive=False):
    # Create acquisition model
    acquisition_model = sirtf.AcquisitionModel(image.shape, image.voxel_size)

    # Create acquisition data
    acquisition_data = sirtf.AcquisitionData(image.shape, image.voxel_size)

    # Create filter
    filter = sirtf.Filter(image.shape, image.voxel_size)

    # Create initial image estimate
    initial_image_estimate = sirtf.Image(image.shape, image.voxel_size)

    # Create prior
    prior = sirtf.Prior(image.shape, image.voxel_size)

    # Create objective function
    objective_function = sirtf.ObjectiveFunction(image.shape, image.voxel_size)

    # Perform OSMAPOSL reconstruction
    for i in range(num_subsets):
        for j in range(num_sub_iterations):
            # Perform one step of OSMAPOSL
            image = sirtf.osmaposl_one_step(image, acquisition_model, acquisition_data, filter, prior, objective_function)

    # Display reconstructed image if non-interactive option is not set
    if not non_interactive:
        sirtf.display_image(image)

def main():
    # Parse command-line options
    raw_data_file = sys.argv[1]
    data_files_path = sys.argv[2]
    num_subsets = int(sys.argv[3])
    num_sub_iterations = int(sys.argv[4])
    reconstruction_engine = sys.argv[5]
    non_interactive = sys.argv[6] == "True"

    # Load raw data
    raw_data = np.load(raw_data_file)

    # Load data files
    data_files = os.listdir(data_files_path)

    # Create acquisition model
    acquisition_model = sirtf.AcquisitionModel(raw_data.shape, raw_data.voxel_size)

    # Create acquisition data
    acquisition_data = sirtf.AcquisitionData(raw_data.shape, raw_data.voxel_size)

    # Create filter
    filter = sirtf.Filter(raw_data.shape, raw_data.voxel_size)

    # Create initial image estimate
    initial_image_estimate = sirtf.Image(raw_data.shape, raw_data.voxel_size)

    # Create prior
    prior = sirtf.Prior(raw_data.shape, raw_data.voxel_size)

    # Create objective function
    objective_function = sirtf.ObjectiveFunction(raw_data.shape, raw_data.voxel_size)

    # Perform reconstruction
    try:
        image = sirtf.reconstruct(raw_data, data_files, acquisition_model, acquisition_data, filter, initial_image_estimate, prior, objective_function, num_subsets, num_sub_iterations, reconstruction_engine)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Display reconstructed image if non-interactive option is not set
    if not non_interactive:
        sirtf.display_image(image)

if __name__ == "__main__":
    main()