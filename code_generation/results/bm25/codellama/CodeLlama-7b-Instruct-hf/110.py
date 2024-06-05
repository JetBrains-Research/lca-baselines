  import sirf
from sirf.reconstruction import create_from_3D_image
from sirf.objective_function import ObjectiveFunction
from sirf.image_data import create_uniform_image
from sirf.acquisition_data import from_acquisition_data
from sirf.objective_function.poisson_log_likelihood import PoissonLogLikelihoodWithLinearModelForMeanAndProjData
from sirf.image_data_processor import TruncateToCylinderProcessor
from sirf.acquisition_data import AcquisitionData
from sirf.test_sti_objective_function import TestSTIRObjectiveFunction
from sirf.image_data import ImageData
from sirf.nifti_image_data import NiftiImageData

def truncate_image(image, radius):
    return TruncateToCylinderProcessor(radius).process(image)

def main(engine, data_file, data_path, num_steps, use_local_opt, verbosity, show_plots):
    # Import the specified engine module
    engine_module = sirf.reconstruction.create_from_3D_image(engine)

    # Process command-line options
    if data_file is None:
        raise ValueError("Data file not specified")
    if data_path is None:
        raise ValueError("Data path not specified")
    if num_steps is None:
        raise ValueError("Number of steps not specified")
    if use_local_opt is None:
        raise ValueError("Use local optimal steepest ascent not specified")
    if verbosity is None:
        raise ValueError("Verbosity not specified")
    if show_plots is None:
        raise ValueError("Show plots not specified")

    # Create an acquisition model
    acquisition_model = engine_module.create_acquisition_model()

    # Read PET acquisition data from the specified file
    acquisition_data = from_acquisition_data(data_file, data_path)

    # Create a filter that zeroes the image outside a cylinder of the same diameter as the image xy-section size
    filter = TruncateToCylinderProcessor(acquisition_data.get_image_xy_size())

    # Create an initial image estimate
    initial_image = create_uniform_image(acquisition_data.get_image_size(), 1.0)

    # Create an objective function of Poisson logarithmic likelihood type compatible with the acquisition data type
    objective_function = PoissonLogLikelihoodWithLinearModelForMeanAndProjData(acquisition_data)

    # Perform the steepest descent steps
    for i in range(num_steps):
        # Compute the gradient of the objective function with respect to the image
        gradient = objective_function.get_gradient(initial_image)

        # Apply the filter to the gradient
        gradient = filter.process(gradient)

        # Update the image estimate using the gradient
        initial_image = initial_image - gradient

        # Print the current image estimate and gradient
        if verbosity > 0:
            print(f"Image estimate: {initial_image}")
            print(f"Gradient: {gradient}")

        # Check for convergence
        if use_local_opt:
            # Compute the local optimal steepest ascent direction
            local_opt_direction = objective_function.get_local_optimal_steepest_ascent_direction(initial_image)

            # Update the image estimate using the local optimal steepest ascent direction
            initial_image = initial_image - local_opt_direction

            # Print the current image estimate and local optimal steepest ascent direction
            if verbosity > 0:
                print(f"Local optimal steepest ascent direction: {local_opt_direction}")
                print(f"Image estimate: {initial_image}")

    # Truncate the image to a cylinder of the same diameter as the image xy-section size
    image = truncate_image(initial_image, acquisition_data.get_image_xy_size())

    # Print the final image estimate
    if verbosity > 0:
        print(f"Final image estimate: {image}")

    # Show plots if requested
    if show_plots:
        # Plot the initial image estimate
        initial_image.show()

        # Plot the final image estimate
        image.show()

if __name__ == "__main__":
    # Parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Reconstruction engine")
    parser.add_argument("--data-file", required=True, help="PET acquisition data file")
    parser.add_argument("--data-path", required=True, help="Path to PET acquisition data files")
    parser.add_argument("--num-steps", required=True, type=int, help="Number of steepest descent steps")
    parser.add_argument("--use-local-opt", required=True, type=bool, help="Use locally optimal steepest ascent")
    parser.add_argument("--verbosity", required=True, type=int, help="Verbosity level")
    parser.add_argument("--show-plots", required=True, type=bool, help="Show plots")
    args = parser.parse_args()

    # Call the main function with the parsed command-line options
    main(args.engine, args.data_file, args.data_path, args.num_steps, args.use_local_opt, args.verbosity, args.show_plots)