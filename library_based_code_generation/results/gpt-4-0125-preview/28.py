```python
import sirf.STIR as stir
import argparse
import matplotlib.pyplot as plt

def osmaposl_reconstruction(image, objective_function, prior, filter, num_subsets, num_subiterations):
    recon = stir.OSMAPOSLReconstruction()
    recon.set_objective_function(objective_function)
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_subiterations)
    if prior is not None:
        recon.set_prior(prior)
    if filter is not None:
        image.apply_filter(filter)
    recon.set_up(image)
    recon.process()
    return recon.get_current_estimate()

def main():
    parser = argparse.ArgumentParser(description="OSMAPOSL Reconstruction")
    parser.add_argument("--raw_data_file", required=True, help="Path to the raw data file")
    parser.add_argument("--data_path", required=True, help="Path to data files")
    parser.add_argument("--num_subsets", type=int, required=True, help="Number of subsets")
    parser.add_argument("--num_subiterations", type=int, required=True, help="Number of sub-iterations")
    parser.add_argument("--reconstruction_engine", choices=['OSEM', 'OSMAPOSL'], required=True, help="Reconstruction engine")
    parser.add_argument("--disable_plots", action='store_true', help="Disable plots")
    args = parser.parse_args()

    try:
        # Load data
        data = stir.AcquisitionData(args.raw_data_file)

        # Create acquisition model
        acq_model = stir.AcquisitionModelUsingRayTracingMatrix()
        acq_model.set_num_tangential_LORs(5)

        # Create initial image estimate
        image = data.create_uniform_image(1.0)

        # Create filter (optional, for demonstration)
        filter = stir.TruncateToCylinderProcessor()

        # Create prior (optional, for demonstration)
        prior = stir.QuadraticPrior()
        prior.set_penalisation_factor(0.5)

        # Create objective function
        if args.reconstruction_engine == 'OSMAPOSL':
            objective_function = stir.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
            objective_function.set_acquisition_model(acq_model)
            objective_function.set_acquisition_data(data)
        else:
            raise ValueError("Unsupported reconstruction engine")

        # Perform reconstruction
        reconstructed_image = osmaposl_reconstruction(image, objective_function, prior, filter, args.num_subsets, args.num_subiterations)

        # Display result
        if not args.disable_plots:
            reconstructed_image.show(title="Reconstructed Image")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
```