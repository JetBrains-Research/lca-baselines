 Here is a simplified version of the requested Python code using the SIRF library. Please note that this is a basic implementation and might need adjustments based on your specific requirements. Also, you may need to install the SIRF library using `pip install sirdar` before running the code.

```python
import sys
import argparse
import sirdar
import numpy as np

def create_acquisition_model(data_path):
    # Load the sinogram data
    sinogram = sirdar.Sinogram.from_file(data_path)

    # Create the acquisition model
    model = sirdar.AcquisitionModel(sinogram)
    return model

def create_filter(filter_type):
    if filter_type == 'shepp_logan':
        return sirdar.SheppLoganFilter()
    elif filter_type == 'ram-lak':
        return sirdar.RamLakFilter()
    else:
        raise ValueError(f"Invalid filter type: {filter_type}")

def create_prior(prior_type):
    if prior_type == 'gaussian':
        return sirdar.GaussianPrior()
    elif prior_type == 'lambertian':
        return sirdar.LambertianPrior()
    else:
        raise ValueError(f"Invalid prior type: {prior_type}")

def create_objective_function(data, filter, prior):
    return sirdar.ObjectiveFunction(data, filter, prior)

def osmap_osl(image, objective_function, prior, filter, n_subsets, n_sub_iterations):
    # Initialize the OSMAPOSL algorithm
    osmap_osl_algorithm = sirdar.OSMAPOSL(n_subsets, n_sub_iterations)

    # Run the algorithm
    reconstructed_image = osmap_osl_algorithm.run(image, objective_function, prior, filter)
    return reconstructed_image

def main(args):
    data_path = args.data
    filter_type = args.filter
    prior_type = args.prior
    n_subsets = int(args.n_subsets)
    n_sub_iterations = int(args.n_sub_iterations)
    reconstruction_engine = args.engine
    plot = not args.no_plot

    # Create the acquisition model
    model = create_acquisition_model(data_path)

    # Create the filter
    filter = create_filter(filter_type)

    # Create the prior
    prior = create_prior(prior_type)

    # Create the objective function
    data = model.data
    objective_function = create_objective_function(data, filter, prior)

    # Initialize the image
    image = np.zeros(data.shape)

    # Run the reconstruction algorithm
    reconstructed_image = osmap_osl(image, objective_function, prior, filter, n_subsets, n_sub_iterations)

    if not plot:
        print(f"Reconstructed image (without plot):\n{reconstructed_image}")
    else:
        from matplotlib import pyplot as plt
        plt.imshow(reconstructed_image, cmap='gray')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to the raw data file")
    parser.add_argument("--filter", default="shepp_logan", choices=["shepp_logan", "ram-lak"], help="Filter type (default: shepp_logan)")
    parser.add_argument("--prior", default="gaussian", choices=["gaussian", "lambertian"], help="Prior type (default: gaussian)")
    parser.add_argument("--n_subsets", type=int, default=10, help="Number of subsets (default: 10)")
    parser.add_argument("--n_sub_iterations", type=int, default=10, help="Number of sub-iterations per subset (default: 10)")
    parser.add_argument("--engine", default="sirdar", choices=["sirdar", "em"], help="Reconstruction engine (default: sirdar)")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()
    main(args)
```

This code defines a main function that handles command-line arguments and calls the `osmap_osl` function to perform the reconstruction. The `osmap_osl` function takes an image, objective function, prior, filter, number of subsets, and number of sub-iterations as parameters and returns the reconstructed image. The main function also creates the acquisition model, filter, prior, and objective function. It handles errors by raising exceptions when invalid options are provided.