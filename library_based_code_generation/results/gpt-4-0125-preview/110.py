```python
import argparse
import numpy as np
from sirf.Utilities import examples_data_path
import sys

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Steepest Ascent for Poisson Log-likelihood Maximization")
    parser.add_argument("--engine", type=str, required=True, help="Reconstruction engine (e.g., STIR, Gadgetron)")
    parser.add_argument("--raw_data_file", type=str, required=True, help="Path to the raw data file")
    parser.add_argument("--data_path", type=str, default=examples_data_path('PET'), help="Path to data files")
    parser.add_argument("--steps", type=int, default=5, help="Number of steepest descent steps")
    parser.add_argument("--local_optimum", action='store_true', help="Use locally optimal steepest ascent")
    parser.add_argument("--verbosity", type=int, default=0, help="Verbosity level")
    parser.add_argument("--show_plots", action='store_true', help="Whether to show plots or not")
    args = parser.parse_args()

    try:
        # Import the specified engine module
        if args.engine == "STIR":
            from sirf.STIR import AcquisitionModel, AcquisitionData, ImageData, ObjectiveFunction
        elif args.engine == "Gadgetron":
            from sirf.Gadgetron import AcquisitionModel, AcquisitionData, ImageData, ObjectiveFunction
        else:
            raise ValueError("Unsupported engine. Please use 'STIR' or 'Gadgetron'.")

        # Read PET acquisition data
        ad = AcquisitionData(args.raw_data_file)

        # Create an acquisition model
        am = AcquisitionModel()
        am.set_up(ad)

        # Create a filter that zeroes the image outside a cylinder
        def truncate_image(image):
            size = image.dimensions()
            xy_size = size[0:2]
            radius = np.min(xy_size) / 2
            for z in range(size[2]):
                for y in range(size[1]):
                    for x in range(size[0]):
                        if np.sqrt((x - size[0] / 2) ** 2 + (y - size[1] / 2) ** 2) > radius:
                            image[z, y, x] = 0
            return image

        # Create an initial image estimate
        initial_image = ad.create_uniform_image(1.0)

        # Create an objective function
        obj_fun = ObjectiveFunction()
        obj_fun.set_acquisition_model(am)
        obj_fun.set_acquisition_data(ad)

        # Perform the steepest descent steps
        current_image = initial_image
        for step in range(args.steps):
            gradient = obj_fun.get_subset_gradient(current_image)
            if args.local_optimum:
                # Adjust step size for locally optimal steepest ascent
                step_size = 1 / (step + 1)
            else:
                step_size = 1
            current_image += gradient * step_size
            current_image = truncate_image(current_image)
            if args.verbosity > 0:
                print(f"Step {step + 1}: Image updated.")

        if args.show_plots:
            current_image.show(title="Final Image")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
```