 Here is a basic structure of the Python code using SIRF library for the OSMAPOSL reconstruction algorithm. Please note that this is a simplified version and might need adjustments based on your specific requirements.

```python
import sys
import sirf
import numpy as np

def create_objective_function(data_handle_vector, filter, max_sigmas):
    return sirf.PoissonLogLikelihoodWithLinearModelForMeanAndProjData(data_handle_vector, filter, max_sigmas)

def create_prior(filter, max_sigmas):
    return sirf.TestSTIRObjectiveFunction(filter, max_sigmas)

def reconstruction(image, objective_function, prior, filter, num_subsets, num_sub_iterations):
    # Create acquisition model
    acquisition_model = sirf.AcquisitionModel(number_of_readouts=objective_function.number_of_readouts(),
                                               number_of_acquisitions=objective_function.number_of_acquisitions())

    # Create acquisition data
    acquisition_data = sirf.AcquisitionData(acquisition_model)

    # Create filter
    acquisition_data.set_filter(filter)

    # Create initial image estimate
    initial_image_estimate = sirf.ImageData(image.field_of_view(), image.shape)
    initial_image_estimate.set_data(image.flatten())

    # Create objective function
    objective_function_obj = create_objective_function(acquisition_data.data_handle_vector(), filter, max_sigmas)

    # Create prior
    prior_obj = create_prior(filter, max_sigmas)

    # Perform OSMAPOSL reconstruction
    reconstructed_image = sirf.osmaposl(initial_image_estimate, objective_function_obj, prior_obj,
                                        num_subsets, num_sub_iterations)

    return reconstructed_image

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="Path to the raw data file")
    parser.add_argument("--subsets", type=int, default=1, help="Number of subsets")
    parser.add_argument("--sub_iterations", type=int, default=1, help="Number of sub-iterations")
    parser.add_argument("--engine", choices=["sirf", "gf"], default="sirf", help="Reconstruction engine")
    parser.add_argument("--no_plot", action="store_true", help="Disable plots")

    args = parser.parse_args()

    # Load data
    data = sirf.load_data(args.data_file)

    # Set filter
    filter = sirf.ButterworthFilter(number_of_sigmas=args.subsets)

    # Set maximum number of sigmas
    filter.set_maximum_number_of_sigmas(args.subsets)

    # Set objective function
    objective_function = create_objective_function(data.acquisition_data.data_handle_vector(), filter, args.subsets)

    # Set prior
    prior = create_prior(filter, args.subsets)

    # Perform reconstruction
    reconstructed_image = reconstruction(data.image_data.image(), objective_function, prior, filter, args.subsets, args.sub_iterations)

    if not args.no_plot:
        sirf.display_image(reconstructed_image)

if __name__ == "__main__":
    main()
```

This code defines a `reconstruction` function that takes an image, objective function, prior, filter, number of subsets, and number of sub-iterations as parameters and performs the OSMAPOSL reconstruction. The `main` function handles command-line options and calls the `reconstruction` function. It also loads the data, sets the filter, objective function, and prior, and displays the reconstructed image if the non-interactive option is not set. Error handling is not included in this example, but you can add it according to your needs.