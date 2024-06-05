```python
import sys
import numpy as np
from sirf.Utilities import examples_data_path
from sirf.STIR import ImageData, AcquisitionData, TruncateToCylinderProcessor, PoissonLogLikelihoodWithLinearModelForMeanAndProjData, read_from_file, create_uniform_image

def truncate_image(image):
    """Truncate the image outside a cylinder of the same diameter as the image xy-section size."""
    processor = TruncateToCylinderProcessor()
    processor.apply(image)

def main(engine, raw_data_file, path_to_data_files, num_steps, use_local_optimum, verbosity, show_plots):
    try:
        # Import the specified engine module dynamically
        if engine not in ['STIR']:
            raise ValueError("Unsupported engine. Currently, only 'STIR' is supported.")
        # Process command-line options
        raw_data_file = path_to_data_files + '/' + raw_data_file

        # Read PET acquisition data
        acq_data = read_from_file(raw_data_file)

        # Create an initial image estimate
        image_size = (128, 128, 64)  # Example size, adjust based on actual data
        voxel_size = (2.5, 2.5, 2.5)  # Example voxel size in mm, adjust based on actual data
        initial_image = create_uniform_image(image_size, voxel_size)
        truncate_image(initial_image)

        # Create an objective function
        obj_fun = PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
        obj_fun.set_acquisition_data(acq_data)
        obj_fun.set_up(initial_image)

        # Perform the steepest descent steps
        for step in range(num_steps):
            current_estimate = obj_fun.get_backprojection_of_acquisition_ratio(acq_data)
            truncate_image(current_estimate)
            if use_local_optimum:
                # Modify this part to implement locally optimal steepest ascent
                pass
            # Update the image estimate (this is a simplified update step)
            initial_image += current_estimate
            truncate_image(initial_image)
            if verbosity:
                print(f"Step {step+1}, Objective Function Value: {obj_fun.value(initial_image)}")

        if show_plots:
            initial_image.show(title="Final Estimate")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python script.py <engine> <raw_data_file> <path_to_data_files> <num_steps> <use_local_optimum> <verbosity> <show_plots>")
        sys.exit(1)
    engine = sys.argv[1]
    raw_data_file = sys.argv[2]
    path_to_data_files = sys.argv[3]
    num_steps = int(sys.argv[4])
    use_local_optimum = sys.argv[5].lower() in ['true', '1', 't', 'y', 'yes']
    verbosity = int(sys.argv[6])
    show_plots = sys.argv[7].lower() in ['true', '1', 't', 'y', 'yes']

    main(engine, raw_data_file, path_to_data_files, num_steps, use_local_optimum, verbosity, show_plots)
```