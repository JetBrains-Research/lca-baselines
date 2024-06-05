```python
import sys
import argparse
import sirf.STIR as stir
from sirf.Utilities import examples_data_path

def osmaposl_reconstruction(image, objective_function, prior, filter, num_subsets, num_subiterations):
    try:
        recon = stir.OSMAPOSLReconstruction()
        recon.set_objective_function(objective_function)
        if prior is not None:
            recon.set_prior(prior)
        recon.set_num_subsets(num_subsets)
        recon.set_num_subiterations(num_subiterations)
        if filter is not None:
            image.apply_filter(filter)
        recon.set_up(image)
        recon.process()
        return recon.get_current_estimate()
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        sys.exit(1)

def main(raw_data_file, data_path, num_subsets, num_subiterations, reconstruction_engine, disable_plots):
    try:
        # Load acquisition data
        acq_data = stir.AcquisitionData(raw_data_file)

        # Create initial image estimate
        image = acq_data.create_uniform_image(1.0)

        # Create acquisition model
        acq_model = stir.AcquisitionModelUsingRayTracingMatrix()
        acq_model.set_up(acq_data, image)

        # Create objective function
        obj_fun = stir.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
        obj_fun.set_acquisition_data(acq_data)
        obj_fun.set_acquisition_model(acq_model)

        # Optionally, set up a prior
        prior = None  # Example: stir.QuadraticPrior() or any other prior

        # Optionally, set up a filter
        filter = None  # Example: stir.SeparableGaussianImageFilter()

        # Perform reconstruction
        reconstructed_image = osmaposl_reconstruction(image, obj_fun, prior, filter, num_subsets, num_subiterations)

        # Display the reconstructed image if plots are not disabled
        if not disable_plots:
            reconstructed_image.show(title="Reconstructed Image")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSMAPOSL Reconstruction using SIRF")
    parser.add_argument("--raw_data_file", type=str, required=True, help="Path to the raw data file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data files")
    parser.add_argument("--num_subsets", type=int, required=True, help="Number of subsets")
    parser.add_argument("--num_subiterations", type=int, required=True, help="Number of sub-iterations")
    parser.add_argument("--reconstruction_engine", type=str, choices=['STIR'], required=True, help="Reconstruction engine")
    parser.add_argument("--disable_plots", action='store_true', help="Disable plotting of results")

    args = parser.parse_args()

    main(args.raw_data_file, args.data_path, args.num_subsets, args.num_subiterations, args.reconstruction_engine, args.disable_plots)
```