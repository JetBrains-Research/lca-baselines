```python
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Estimate randoms from a list mode file and compare with original delayed coincidences.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data files.")
    parser.add_argument("--listmode-file", type=str, required=True, help="Listmode file.")
    parser.add_argument("--sinograms-prefix", type=str, required=True, help="Sinograms file prefix.")
    parser.add_argument("--randoms-file", type=str, required=True, help="Randoms file.")
    parser.add_argument("--raw-data-template", type=str, required=True, help="Raw data template.")
    parser.add_argument("--scanning-time-interval", type=int, nargs=2, required=True, help="Scanning time interval.")
    parser.add_argument("--reconstruction-engine", type=str, required=True, help="Reconstruction engine.")
    parser.add_argument("--acquisition-data-storage", type=str, required=True, help="Acquisition data storage scheme.")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode.")
    args = parser.parse_args()

    try:
        # Importing necessary modules for processing
        from sirf.STIR import ListmodeToSinograms, AcquisitionData

        # Setting up the listmode-to-sinograms converter
        converter = ListmodeToSinograms()
        converter.set_input(args.listmode_file)
        converter.set_output_prefix(args.sinograms_prefix)
        converter.set_template(args.raw_data_template)
        converter.set_time_interval(args.scanning_time_interval[0], args.scanning_time_interval[1])
        converter.flag_on("store_delayeds")

        # Processing the data
        converter.process()
        sinograms = converter.get_output()
        acquisition_data = AcquisitionData(args.randoms_file)

        # Estimating randoms from delayeds
        randoms = sinograms.estimate_randoms()

        # Writing estimated randoms to a file
        randoms.write(args.randoms_file)

        # Copying acquisition data into Python arrays
        acq_array = acquisition_data.as_array()
        randoms_array = randoms.as_array()

        # Printing out information
        print(f"Acquisition data dimensions: {acq_array.shape}")
        print(f"Total number of delayed coincidences: {np.sum(acq_array)}")
        print(f"Total number of estimated randoms: {np.sum(randoms_array)}")
        print(f"Max value in acquisition data: {np.max(acq_array)}")
        print(f"Max value in estimated randoms: {np.max(randoms_array)}")

        if not args.non_interactive:
            # Displaying a single sinogram
            plt.figure()
            plt.imshow(acq_array[:, acq_array.shape[1] // 2, :], aspect='auto')
            plt.title("Single Sinogram Slice")
            plt.colorbar()
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```