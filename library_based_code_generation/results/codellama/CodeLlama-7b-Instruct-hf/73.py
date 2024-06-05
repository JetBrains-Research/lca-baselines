  import os
import sys
import argparse
import numpy as np
from sirf import SIRF

def main():
    # Parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to data files")
    parser.add_argument("--listmode-file", required=True, help="Path to listmode file")
    parser.add_argument("--output-prefix", required=True, help="Output file prefix")
    parser.add_argument("--raw-data-template", required=True, help="Raw data template")
    parser.add_argument("--scanning-time-interval", required=True, help="Scanning time interval to convert")
    parser.add_argument("--reconstruction-engine", required=True, help="Reconstruction engine")
    parser.add_argument("--acquisition-data-storage-scheme", required=True, help="Acquisition data storage scheme")
    parser.add_argument("--disable-interactive-mode", action="store_true", help="Disable interactive mode")
    args = parser.parse_args()

    # Set acquisition data storage scheme
    acquisition_data_storage_scheme = args.acquisition_data_storage_scheme

    # Read acquisition data template
    raw_data_template = args.raw_data_template

    # Create listmode-to-sinograms converter object
    converter = SIRF.ListmodeToSinogramsConverter()

    # Set input, output and template files
    converter.setInputFile(args.listmode_file)
    converter.setOutputFilePrefix(args.output_prefix)
    converter.setTemplateFile(raw_data_template)

    # Set time interval
    converter.setTimeInterval(args.scanning_time_interval)

    # Set some flags
    converter.setUseInteractiveMode(not args.disable_interactive_mode)

    # Set up converter
    converter.setup()

    # Perform conversion
    converter.convert()

    # Get access to sinograms
    sinograms = converter.getSinograms()

    # Copy acquisition data into a Python array
    acquisition_data = np.array(sinograms.getAcquisitionData())

    # Print acquisition data dimensions
    print("Acquisition data dimensions:", acquisition_data.shape)

    # Optionally show a 2D array of the acquisition data
    if args.show_acquisition_data:
        print("Acquisition data:")
        print(acquisition_data)

    # Estimate randoms
    randoms = sinograms.estimateRandoms()

    # Convert randoms to an array
    randoms_array = np.array(randoms)

    # Optionally show a 2D array of the randoms
    if args.show_randoms:
        print("Randoms:")
        print(randoms_array)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")