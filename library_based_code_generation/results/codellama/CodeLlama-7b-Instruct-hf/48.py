  import os
import numpy as np
from smlmlib import ListmodeToSinogramsConverter
from smlmlib import MaximumLikelihoodEstimator

def main(args):
    # Set up the listmode-to-sinograms converter
    converter = ListmodeToSinogramsConverter(
        input_file=args.listmode_file,
        output_file_prefix=args.sinograms_file_prefix,
        template_file=args.raw_data_template,
        scanning_time_interval=args.scanning_time_interval,
        reconstruction_engine=args.reconstruction_engine,
        acquisition_data_storage_scheme=args.acquisition_data_storage_scheme,
        non_interactive_mode=args.non_interactive_mode
    )

    # Process the data and get access to the sinograms
    converter.process_data()
    sinograms = converter.get_sinograms()

    # Estimate the randoms from the delayeds via Maximum Likelihood estimation
    estimator = MaximumLikelihoodEstimator(sinograms)
    randoms = estimator.estimate_randoms()

    # Write the estimated randoms to a file
    np.savetxt(args.randoms_file, randoms)

    # Copy the acquisition data into Python arrays and print out the acquisition data dimensions, total number of delayed coincidences, and estimated randoms
    acquisition_data = converter.get_acquisition_data()
    acquisition_data_dimensions = acquisition_data.shape
    total_number_of_delayed_coincidences = np.sum(acquisition_data)
    estimated_randoms = np.sum(randoms)
    print(f"Acquisition data dimensions: {acquisition_data_dimensions}")
    print(f"Total number of delayed coincidences: {total_number_of_delayed_coincidences}")
    print(f"Estimated randoms: {estimated_randoms}")

    # If not in non-interactive mode, display a single sinogram
    if not args.non_interactive_mode:
        converter.display_sinogram(0)

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--listmode_file", required=True, help="Path to the listmode file")
    parser.add_argument("--sinograms_file_prefix", required=True, help="Prefix for the sinograms file")
    parser.add_argument("--raw_data_template", required=True, help="Path to the raw data template file")
    parser.add_argument("--scanning_time_interval", required=True, help="Scanning time interval")
    parser.add_argument("--reconstruction_engine", required=True, help="Reconstruction engine")
    parser.add_argument("--acquisition_data_storage_scheme", required=True, help="Acquisition data storage scheme")
    parser.add_argument("--non_interactive_mode", action="store_true", help="Non-interactive mode")
    parser.add_argument("--randoms_file", required=True, help="Path to the randoms file")
    args = parser.parse_args()

    # Set up the listmode-to-sinograms converter
    converter = ListmodeToSinogramsConverter(
        input_file=args.listmode_file,
        output_file_prefix=args.sinograms_file_prefix,
        template_file=args.raw_data_template,
        scanning_time_interval=args.scanning_time_interval,
        reconstruction_engine=args.reconstruction_engine,
        acquisition_data_storage_scheme=args.acquisition_data_storage_scheme,
        non_interactive_mode=args.non_interactive_mode
    )

    # Process the data and get access to the sinograms
    converter.process_data()
    sinograms = converter.get_sinograms()

    # Estimate the randoms from the delayeds via Maximum Likelihood estimation
    estimator = MaximumLikelihoodEstimator(sinograms)
    randoms = estimator.estimate_randoms()

    # Write the estimated randoms to a file
    np.savetxt(args.randoms_file, randoms)

    # Copy the acquisition data into Python arrays and print out the acquisition data dimensions, total number of delayed coincidences, and estimated randoms
    acquisition_data = converter.get_acquisition_data()
    acquisition_data_dimensions = acquisition_data.shape
    total_number_of_delayed_coincidences = np.sum(acquisition_data)
    estimated_randoms = np.sum(randoms)
    print(f"Acquisition data dimensions: {acquisition_data_dimensions}")
    print(f"Total number of delayed coincidences: {total_number_of_delayed_coincidences}")
    print(f"Estimated randoms: {estimated_randoms}")

    # If not in non-interactive mode, display a single sinogram
    if not args.non_interactive_mode:
        converter.display_sinogram(0)

    # Handle any errors that occur during execution
    except Exception as e:
        print(f"Error: {e}")