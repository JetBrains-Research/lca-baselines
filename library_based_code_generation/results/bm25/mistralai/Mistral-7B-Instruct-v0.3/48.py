 Here is a Python script that follows your instructions. I've used the `argparse` module for command-line options and the `petmr` library for the necessary APIs. Please make sure to install the `petmr` library before running the script.

```python
import argparse
import petmr

def main(args):
    # Initialize the argument parser
    parser = argparse.ArgumentParser()

    # Add command-line options
    parser.add_argument('--data_path', type=str, required=True, help='Path to data files')
    parser.add_argument('--listmode_file', type=str, required=True, help='Listmode file')
    parser.add_argument('--sinograms_prefix', type=str, required=True, help='Sinograms file prefix')
    parser.add_argument('--randoms_file', type=str, required=True, help='Randoms file')
    parser.add_argument('--raw_data_template', type=str, required=True, help='Raw data template')
    parser.add_argument('--scanning_time_interval', type=float, required=True, help='Scanning time interval')
    parser.add_argument('--reconstruction_engine', type=str, required=True, help='Reconstruction engine')
    parser.add_argument('--acquisition_data_storage_scheme', type=str, required=True, help='Acquisition data storage scheme')
    parser.add_argument('--non_interactive', action='store_true', help='Run in non-interactive mode')

    # Parse the arguments
    args = parser.parse_args()

    # Set up the ListmodeToSinograms converter
    converter = petmr.ListmodeToSinograms(
        input_file=args.listmode_file,
        output_file_prefix=args.sinograms_prefix,
        template_file=args.raw_data_template
    )
    converter.set_time_interval(args.scanning_time_interval)
    converter.set_delayed_coincidences_storage(True)

    # Process the data
    converter.process()

    # Get access to the sinograms and estimate the randoms
    sinograms = converter.get_sinograms()
    randoms_estimator = petmr.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
    randoms = randoms_estimator.estimate_randoms(sinograms)

    # Write the estimated randoms to a file
    with open(args.randoms_file, 'w') as f:
        for row in randoms:
            f.write(' '.join(map(str, row)) + '\n')

    # Copy the acquisition data into Python arrays
    acquisition_data = converter.get_acquisition_data()

    # Print out the acquisition data dimensions, total number of delayed coincidences and estimated randoms, and max values
    print(f"Acquisition data dimensions: {acquisition_data.shape}")
    print(f"Total number of delayed coincidences: {converter.get_total_delayed_coincidences()}")
    print(f"Total number of estimated randoms: {len(randoms)}")
    print(f"Max values in acquisition data: {acquisition_data.max()}")

    # If not in non-interactive mode, display a single sinogram
    if not args.non_interactive:
        sinogram = sinograms[0]
        print("Sinogram:")
        for row in sinogram:
            print(' '.join(map(str, row)))

if __name__ == "__main__":
    main(petmr_data_path)
```

This script assumes that the `petmr_data_path` variable is defined before the `if __name__ == "__main__":` line and points to the directory containing the data files. You can set this variable according to your data organization.