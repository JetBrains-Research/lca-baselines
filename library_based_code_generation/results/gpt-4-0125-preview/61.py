```python
import argparse
import sys
import os
try:
    import sirf.Gadgetron as gt
except ImportError:
    print("SIRF Gadgetron module not found.")
    sys.exit()

def conjugate_gradient_reconstruction(acq_data, trajectory, show_plots=False):
    # Compute coil sensitivity maps
    csm = gt.CoilSensitivityData()
    csm.calculate(acq_data)

    # Set up acquisition model
    acq_model = gt.AcquisitionModel(acq_data, csm)
    acq_model.set_up(acq_data)

    # Perform backward projection to get an initial estimate
    complex_img = acq_model.backward(acq_data)

    # Define symmetrical operator for cg-optimisation
    sym_op = gt.SymmetricalOperator(acq_model)

    # Perform iterative reconstruction using Conjugate Gradient method
    recon = gt.ConjugateGradientReconstructor()
    recon.set_operator(sym_op)
    recon.set_max_iterations(10)
    recon.set_input(complex_img)
    recon.process()
    result = recon.get_output()

    if show_plots:
        result.show(title="Reconstructed Image")

    return result

def main():
    parser = argparse.ArgumentParser(description="Iterative reconstruction with radial phase encoding (RPE) data using SIRF.")
    parser.add_argument("--raw_data_file", type=str, required=True, help="Path to the raw k-space data file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data files.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for simulated data.")
    parser.add_argument("--engine", type=str, choices=['Gadgetron'], default='Gadgetron', help="Reconstruction engine.")
    parser.add_argument("--run_reconstruction", action='store_true', help="Run the reconstruction if non-cartesian code was compiled.")
    parser.add_argument("--trajectory", type=str, choices=['cartesian', 'radial', 'goldenangle', 'grpe'], required=True, help="Type of trajectory.")
    parser.add_argument("--show_plots", action='store_true', help="Whether to show plots.")
    args = parser.parse_args()

    # Locate the k-space raw data file
    raw_data_file = os.path.join(args.data_path, args.raw_data_file)
    if not os.path.exists(raw_data_file):
        print(f"Error: Raw data file {raw_data_file} not found.")
        sys.exit()

    # Read acquisition data from an HDF file
    try:
        acq_data = gt.AcquisitionData(raw_data_file)
    except Exception as e:
        print(f"Error reading acquisition data: {e}")
        sys.exit()

    # Pre-process acquisition data
    preprocessed_data = gt.preprocess_acquisition_data(acq_data)

    # Set the trajectory
    if args.trajectory != 'cartesian':
        preprocessed_data.sort()
        if args.trajectory == 'radial' or args.trajectory == 'goldenangle':
            preprocessed_data = gt.set_radial_trajectory(preprocessed_data)
        elif args.trajectory == 'grpe':
            preprocessed_data = gt.set_grpe_trajectory(preprocessed_data)

    # Perform the reconstruction if the relevant option is set
    if args.run_reconstruction:
        reconstructed_image = conjugate_gradient_reconstruction(preprocessed_data, args.trajectory, args.show_plots)
        reconstructed_image.write(args.output_file)

if __name__ == "__main__":
    main()
```