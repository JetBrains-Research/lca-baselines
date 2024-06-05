



 Here is the Python code that performs a few steps of steepest ascent for the maximization of Poisson log-likelihood objective function using subset gradients:

```
import argparse
import logging
import os
import sys

import numpy as np

import sirf.Gadgetron as pMR
import sirf.Reg as reg
import sirf.STIR as pet
import sirf.Utilities as util


def truncate_image(image, cylinder_diameter):
    """Truncate the image outside a cylinder of the same diameter as the image xy-section size."""
    image_size = image.as_array().shape
    image_center = (image_size[0] // 2, image_size[1] // 2)
    radius = cylinder_diameter // 2

    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if ((i - image_center[0]) ** 2 + (j - image_center[1]) ** 2) > (radius ** 2):
                image.fill(0, [i, j, ':'])

    return image


def main(args):
    # Create an acquisition model
    acq_model = pMR.AcquisitionModel(args.reconstruction_engine)

    # Read PET acquisition data from the specified file
    acq_data = pet.AcquisitionData(args.raw_data_file)

    # Create a filter that zeroes the image outside a cylinder of the same diameter as the image xy-section size
    filter = util.CylindricalFilter(acq_data.geometry, args.cylinder_diameter)

    # Create an initial image estimate
    image = util.ImageData(acq_data.geometry.get_ImageGeometry())
    image.fill(0)

    # Create an objective function of Poisson logarithmic likelihood type compatible with the acquisition data type
    objective_function = reg.PoissonLogLikelihood(acq_data, acq_model, image)

    # Perform the steepest descent steps
    for i in range(args.num_descent_steps):
        try:
            logging.info(f'Performing steepest descent step {i + 1}...')
            objective_function.run(args.num_subsets, args.use_locally_optimal_gradient)
            image = objective_function.get_x()
            image = truncate_image(image, args.cylinder_diameter)

            if args.verbosity >= 1:
                logging.info(f'Objective function value: {objective_function.get_value()}')

            if args.show_plots:
                image.show()
        except Exception as e:
            logging.error(f'Error during steepest descent step {i + 1}: {e}')
            break


if __name__ == '__main__':
    # Parse command-line options
    parser = argparse.ArgumentParser(description='Steepest descent for Poisson log-likelihood objective function')
    parser.add_argument('--reconstruction-engine', type=str, required=True, help='Reconstruction engine module')
    parser.add_argument('--raw-data-file', type=str, required=True, help='Path to raw data file')
    parser.add_argument('--cylinder-diameter', type=int, required=True, help='Diameter of cylinder for truncation')
    parser.add_argument('--num-descent-steps', type=int, default=1, help='Number of steepest descent steps')
    parser.add_argument('--num-subsets', type=int, default=1, help='Number of subsets for subset gradients')
    parser.add_argument('--use-locally-optimal-gradient', action='store_true', help='Use locally optimal gradient')
    parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level')
    parser.add_argument('--show-plots', action='store_true', help='Show plots')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO if args.verbosity >= 1 else logging.WARNING)

    # Import the specified reconstruction engine module
    sys.path.append(os.path.dirname(args.reconstruction_engine))
    importlib.import_module(os.path.basename(args.reconstruction_engine))

    # Run the main function
    main(args)
```

This code performs a few steps of steepest ascent for the maximization of Poisson log-likelihood objective function using subset gradients. It allows the user to specify the reconstruction engine, raw data file, path to data files, number of steepest descent steps, whether to use locally optimal steepest ascent, verbosity, and whether to show plots or not. The code imports the specified engine module from the sirf library, processes the command-line options, and defines a function to truncate the image. The main function creates an acquisition model, reads PET acquisition data from the specified file, creates a filter that zeroes the image outside a cylinder of the same diameter as the image xy-section size, creates an initial image estimate, creates an objective function of Poisson logarithmic likelihood type compatible with the acquisition data type, and performs the steepest descent steps. If anything goes wrong, the code catches and displays the error information.