


```python
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from sirf import STIR
from sirf import AcquisitionData, AcquisitionModel, PoissonLogLikelihoodWithLinearModelForMeanAndProjData, \
    CGSolver, SymmetrisedGradientOperator, ImageData, DataContainer, DataProcessor, \
    CoilSensitivityData, DataHandleVector, examples_data_path, mr_data_path, petmr_data_path


def main():
    parser = argparse.ArgumentParser(description='Reconstruct raw RPE data using SIRF')
    parser.add_argument('--raw_data_file', dest='raw_data_file', type=str, required=True,
                        help='Path to the raw data file')
    parser.add_argument('--data_path', dest='data_path', type=str, required=True,
                        help='Path to the data files')
    parser.add_argument('--output_file', dest='output_file', type=str, required=True,
                        help='Output file for simulated data')
    parser.add_argument('--reconstruction_engine', dest='reconstruction_engine', type=str, required=True,
                        help='Reconstruction engine to use')
    parser.add_argument('--run_reconstruction', dest='run_reconstruction', type=bool, required=True,
                        help='Whether to run the reconstruction if non-cartesian code was compiled')
    parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, required=True,
                        help='Trajectory type (cartesian, radial, goldenangle or grpe)')
    parser.add_argument('--show_plots', dest='show_plots', type=bool, required=True,
                        help='Whether to show plots')

    args = parser.parse_args()

    # Import the necessary engine module from the SIRF library
    engine_module = __import__(args.reconstruction_engine)

    # Process the command-line options
    raw_data_file = args.raw_data_file
    data_path = args.data_path
    output_file = args.output_file
    reconstruction_engine = args.reconstruction_engine
    run_reconstruction = args.run_reconstruction
    trajectory_type = args.trajectory_type
    show_plots = args.show_plots

    # Define a symmetrical operator for cg-optimisation
    A = SymmetrisedGradientOperator(engine_module.RegularizationOperator(1),
                                    engine_module.ForwardProjector(args.reconstruction_engine))

    def cg_recon(acq_data, acq_model, niter, nsubiter, tol):
        """Perform Conjugate Gradient reconstruction"""
        # Compute coil sensitivity maps
        CSM = CoilSensitivityData(acq_data)

        # Set up the acquisition model
        acq_model.set_coil_sensitivity_model(CSM)

        # Perform backward projection
        bck_proj = acq_model.backward(acq_data)

        # Initialise the image
        image = bck_proj.clone()
        image.fill(0)

        # Define the log-likelihood function
        loglikelihood = PoissonLogLikelihoodWithLinearModelForMeanAndProjData(acq_model, acq_data, bck_proj)

        # Define the CG solver
        cg_solver = CGSolver(loglikelihood, A, image, niter, nsubiter, tol)

        # Run the CG solver
        cg_solver.run()

        return image

    def main_function():
        """Main function for reconstructing raw RPE data"""
        try:
            # Locate the k-space raw data file
            kspace_data_file = os.path.join(data_path, raw_data_file)

            # Read the acquisition data from an HDF file
            acq_data = AcquisitionData(kspace_data_file)

            # Pre-process the acquisition data
            processor = DataProcessor()
            processor.set_data_trajectory(acq_data)
            processor.sort_data_trajectory()

            # Set the trajectory
            trajectory = processor.get_data_trajectory()

            # Sort the processed acquisition data
            sorted_acq_data = processor.get_processed_data()

            # Perform the reconstruction if the relevant option is set
            if run_reconstruction:
                # Set up the acquisition model
                acq_model = AcquisitionModel(trajectory, reconstruction_engine)

                # Perform the reconstruction
                image = cg_recon(sorted_acq_data, acq_model, 10, 2, 1e-5)

                # Save the reconstructed image
                image.write(output_file)

                # Show plots if specified
                if show_plots:
                    plt.imshow(image.as_array())
                    plt.show()
        except Exception as e:
            print(f'Error during reconstruction: {e}')
            sys.exit(1)

    if __name__ == '__main__':
        main_function()
```

This code defines a main function that performs an iterative reconstruction with radial phase encoding (RPE) data using the SIRF library. It includes command-line options for specifying the raw data file, path to data files, output file for simulated data, reconstruction engine, and whether to run the reconstruction if non-cartesian code was compiled. It also includes an option for specifying the trajectory type (cartesian, radial, goldenangle or grpe) and whether to show plots. The code imports the necessary engine module from the SIRF library based on the specified engine option. It then processes the command-line options and defines a symmetrical operator for cg-optimisation. The code defines a function for performing the Conjugate Gradient method, which includes computing coil sensitivity maps, setting up the acquisition model, performing backward projection, and implementing the iterative reconstruction. Finally, the code defines a main function that locates the k-space raw data file, reads the acquisition data from an HDF file, pre-processes the acquisition data, sets the trajectory, sorts the processed acquisition data, and performs the reconstruction if the relevant option is set. The code handles any errors that occur during execution and prints an appropriate error message.

Please note that this code is for illustrative purposes only and may require modification to fit your specific needs. Additionally, it is important to ensure that you have the necessary permissions and licenses to use the SIRF library and any related data.