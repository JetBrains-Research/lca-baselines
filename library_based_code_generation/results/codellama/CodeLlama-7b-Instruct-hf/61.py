  import os
import sys
import numpy as np
import h5py
from sirf import SIRF
from sirf.reconstruction import ReconstructionEngine
from sirf.reconstruction.cg import ConjugateGradient
from sirf.reconstruction.cg.utils import compute_coil_sensitivity_maps
from sirf.reconstruction.cg.acquisition_model import AcquisitionModel
from sirf.reconstruction.cg.backprojection import Backprojection
from sirf.reconstruction.cg.iterative_reconstruction import IterativeReconstruction
from sirf.reconstruction.cg.utils import sort_acquisition_data

def main():
    # Parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_file", required=True, help="Path to the raw data file")
    parser.add_argument("--data_path", required=True, help="Path to the data files")
    parser.add_argument("--output_file", required=True, help="Path to the output file for simulated data")
    parser.add_argument("--engine", required=True, help="Reconstruction engine to use (e.g. 'cg', 'fbp', 'cs', etc.)")
    parser.add_argument("--trajectory", required=True, help="Trajectory type (e.g. 'cartesian', 'radial', 'goldenangle', etc.)")
    parser.add_argument("--show_plots", action="store_true", help="Show plots during reconstruction")
    parser.add_argument("--run_reconstruction", action="store_true", help="Run the reconstruction if non-cartesian code was compiled")
    args = parser.parse_args()

    # Import the necessary engine module from the SIRF library
    if args.engine == "cg":
        from sirf.reconstruction.cg import ConjugateGradient
    elif args.engine == "fbp":
        from sirf.reconstruction.fbp import FBP
    elif args.engine == "cs":
        from sirf.reconstruction.cs import ConstrainedSensitivity
    else:
        raise ValueError(f"Invalid reconstruction engine: {args.engine}")

    # Define a symmetrical operator for cg-optimisation
    symmetrical_operator = SIRF.SymmetricalOperator(args.trajectory)

    # Define a function for performing the Conjugate Gradient method
    def conjugate_gradient(acquisition_data, symmetrical_operator, show_plots=False):
        # Compute coil sensitivity maps
        coil_sensitivity_maps = compute_coil_sensitivity_maps(acquisition_data)

        # Set up the acquisition model
        acquisition_model = AcquisitionModel(acquisition_data, coil_sensitivity_maps)

        # Perform backward projection
        backprojection = Backprojection(acquisition_model, symmetrical_operator)

        # Implement the iterative reconstruction
        iterative_reconstruction = IterativeReconstruction(backprojection, symmetrical_operator)

        # Perform the conjugate gradient method
        reconstructed_data = conjugate_gradient(iterative_reconstruction, symmetrical_operator, show_plots)

        return reconstructed_data

    # Locate the k-space raw data file
    raw_data_file = os.path.join(args.data_path, args.raw_data_file)

    # Read the acquisition data from an HDF file
    with h5py.File(raw_data_file, "r") as f:
        acquisition_data = f["acquisition_data"][:]

    # Pre-process the acquisition data
    acquisition_data = np.asarray(acquisition_data)

    # Set the trajectory
    trajectory = SIRF.Trajectory(args.trajectory)

    # Sort the processed acquisition data
    sorted_acquisition_data = sort_acquisition_data(acquisition_data, trajectory)

    # Perform the reconstruction if the relevant option is set
    if args.run_reconstruction:
        reconstructed_data = conjugate_gradient(sorted_acquisition_data, symmetrical_operator, show_plots=args.show_plots)

        # Write the reconstructed data to an HDF file
        with h5py.File(args.output_file, "w") as f:
            f.create_dataset("reconstructed_data", data=reconstructed_data)

if __name__ == "__main__":
    main()