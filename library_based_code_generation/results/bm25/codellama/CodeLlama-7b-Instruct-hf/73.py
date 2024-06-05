  Here is the Python code based on the given instruction:
```
import sys
import os
import numpy as np
from sirf import AcquisitionData, AcquisitionDataProcessor, ListmodeToSinograms
from sirf.data_algebra import TestSTIRAcquisitionDataAlgebraFile, TestGadgetronAcquisitionDataAlgebra
from sirf.data_algebra import DataHandleVector, set_data_trajectory, show_3D_array
from sirf.data_algebra import TestSTIRAcquisitionDataAlgebraMemory, petmr_data_path, mr_data_path, examples_data_path

def main():
    # Process command-line options
    data_path = sys.argv[1]
    listmode_file = sys.argv[2]
    output_prefix = sys.argv[3]
    raw_data_template = sys.argv[4]
    scanning_time_interval = float(sys.argv[5])
    reconstruction_engine = sys.argv[6]
    acquisition_data_storage_scheme = sys.argv[7]
    disable_interactive_mode = sys.argv[8] == "True"

    # Import necessary modules
    import sirf
    import sirf.data_algebra

    # Set acquisition data storage scheme
    sirf.data_algebra.set_acquisition_data_storage_scheme(acquisition_data_storage_scheme)

    # Read acquisition data template
    acquisition_data_template = sirf.data_algebra.new_acquisition_data(raw_data_template)

    # Create listmode-to-sinograms converter object
    converter = ListmodeToSinograms(reconstruction_engine)

    # Set input, output, and template files
    converter.set_input_file(listmode_file)
    converter.set_output_file(output_prefix)
    converter.set_template_file(acquisition_data_template)

    # Set time interval
    converter.set_time_interval(scanning_time_interval)

    # Set some flags
    converter.set_disable_interactive_mode(disable_interactive_mode)

    # Set up converter
    converter.setup()

    # Perform conversion
    converter.convert()

    # Get access to sinograms
    sinograms = converter.get_sinograms()

    # Copy acquisition data into a Python array
    acquisition_data = sirf.data_algebra.from_acquisition_data(sinograms)

    # Print acquisition data dimensions
    print(acquisition_data.shape)

    # Optionally show a 2D array of the acquisition data
    if not disable_interactive_mode:
        sirf.data_algebra.show_2D_array(acquisition_data)

    # Estimate randoms
    randoms = converter.estimate_randoms()

    # Convert randoms to an array
    randoms_array = sirf.data_algebra.from_acquisition_data(randoms)

    # Optionally show a 2D array of the randoms
    if not disable_interactive_mode:
        sirf.data_algebra.show_2D_array(randoms_array)

if __name__ == "__main__":
    main()
```
Note that this code assumes that the necessary modules and functions are imported and that the command-line options are processed correctly. Additionally, this code assumes that the acquisition data storage scheme, acquisition data template, and listmode file are provided as command-line options.