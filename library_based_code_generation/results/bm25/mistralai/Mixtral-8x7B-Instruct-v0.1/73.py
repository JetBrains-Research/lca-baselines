 import argparse
import numpy as np
from sirf.IO import AcquisitionData
from sirf.AcquisitionDataProcessor import AcquisitionDataProcessor
from sirf.ListmodeToSinograms import ListmodeToSinograms
from sirf.DataHandleVector import DataHandleVector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("listmode_file")
    parser.add_argument("output_prefix")
    parser.add_argument("raw_data_template")
    parser.add_argument("time_interval")
    parser.add_argument("recon_engine")
    parser.add_argument("acquisition_data_storage_scheme")
    parser.add_argument("--interactive", action="store_false", dest="non_interactive")

    args = parser.parse_args()

    # Set acquisition data storage scheme
    AcquisitionData.set_acquisition_data_storage_scheme(args.acquisition_data_storage_scheme)

    # Create a listmode-to-sinograms converter object
    converter = ListmodeToSinograms()

    # Read the acquisition data template
    template = np.load(args.raw_data_template)

    # Create a data handle vector
    dhv = DataHandleVector()

    # Set the input, output and template files
    dhv.set_input_file(args.listmode_file)
    dhv.set_output_file(args.output_prefix + "_sinograms.h5")
    dhv.set_template_file(args.raw_data_template)

    # Set the time interval
    dhv.set_time_interval(float(args.time_interval))

    # Set some flags
    dhv.set_flag("do_attenuation_correction", 0)
    dhv.set_flag("do_randoms_correction", 1)
    dhv.set_flag("do_norm", 1)

    # Set up the converter
    converter.setup(dhv)

    # Perform the conversion
    converter.convert()

    # Get access to the sinograms
    sinograms = converter.get_output_sinograms()

    # Copy the acquisition data into a Python array
    acquisition_data = np.empty(sinograms.get_acquisition_data_shape(), dtype=sinograms.get_acquisition_data_type())
    sinograms.copy_acquisition_data_to_array(acquisition_data)

    # Print the acquisition data dimensions
    print("Acquisition data dimensions:", acquisition_data.shape)

    # Estimate randoms
    randoms = converter.estimate_randoms()

    # Convert the randoms to an array
    randoms_array = np.empty(randoms.get_acquisition_data_shape(), dtype=randoms.get_acquisition_data_type())
    randoms.copy_acquisition_data_to_array(randoms_array)

    # Optionally show a 2D array of the acquisition data
    if args.non_interactive:
        show_2D_array(acquisition_data)

    # Optionally show a 2D array of the randoms
    if args.non_interactive:
        show_2D_array(randoms_array)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred: ", str(e))