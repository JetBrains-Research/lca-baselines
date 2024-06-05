```python
import sys
import argparse
import numpy as np
import sirf.STIR as stir

def main(args):
    # Set acquisition data storage scheme
    stir.AcquisitionData.set_storage_scheme(args.acquisition_data_storage_scheme)

    # Read the acquisition data template
    template = stir.AcquisitionData(args.raw_data_template)

    # Create a listmode-to-sinograms converter object
    lm2sino = stir.ListmodeToSinograms()

    # Set the input, output, and template files
    lm2sino.set_input(args.listmode_file)
    lm2sino.set_output_prefix(args.output_file_prefix)
    lm2sino.set_template(template)

    # Set the time interval
    lm2sino.set_time_interval(args.scanning_time_interval[0], args.scanning_time_interval[1])

    # Set some flags
    lm2sino.flag_on(stir.ListmodeToSinograms.DO_TIME_OF_FLIGHT)
    lm2sino.flag_off(stir.ListmodeToSinograms.DO_STORE_PROMPTS)

    if not args.interactive_mode:
        lm2sino.set_up()
        lm2sino.process()

        # Get access to the sinograms
        sinograms = lm2sino.get_output()
        sinogram_array = sinograms.as_array()
        print("Sinogram dimensions:", sinogram_array.shape)

        # Optionally show a 2D array of the sinograms
        if args.show_sinograms:
            import matplotlib.pyplot as plt
            plt.imshow(sinogram_array[0, :, :], cmap='gray')
            plt.show()

        # Estimate randoms
        randoms = lm2sino.estimate_randoms()
        randoms_array = randoms.as_array()

        # Optionally show a 2D array of the randoms
        if args.show_randoms:
            plt.imshow(randoms_array[0, :, :], cmap='gray')
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert listmode data to sinograms.")
    parser.add_argument("--path_to_data_files", type=str, required=True)
    parser.add_argument("--listmode_file", type=str, required=True)
    parser.add_argument("--output_file_prefix", type=str, required=True)
    parser.add_argument("--raw_data_template", type=str, required=True)
    parser.add_argument("--scanning_time_interval", type=int, nargs=2, required=True)
    parser.add_argument("--reconstruction_engine", type=str, required=True)
    parser.add_argument("--acquisition_data_storage_scheme", type=str, required=True)
    parser.add_argument("--interactive_mode", action='store_false')
    parser.add_argument("--show_sinograms", action='store_true')
    parser.add_argument("--show_randoms", action='store_true')

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print("An error occurred:", e)
        sys.exit(1)
```