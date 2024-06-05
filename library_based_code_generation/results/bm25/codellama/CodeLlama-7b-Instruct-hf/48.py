  Here is the Python code based on the given instruction:
```
import sys
import os
import numpy as np
from stira.listmode import ListmodeToSinograms
from stira.reconstruction import MaximumLikelihoodEstimator
from stira.data import AcquisitionData
from stira.data import DataHandleVector
from stira.data import AcquisitionDataProcessor

def main():
    # Parse command-line options
    path_to_data_files = sys.argv[1]
    listmode_file = sys.argv[2]
    sinograms_file_prefix = sys.argv[3]
    randoms_file = sys.argv[4]
    raw_data_template = sys.argv[5]
    scanning_time_interval = float(sys.argv[6])
    reconstruction_engine = sys.argv[7]
    acquisition_data_storage_scheme = sys.argv[8]
    non_interactive_mode = sys.argv[9] == "True"

    # Import necessary modules
    import stira.listmode as listmode
    import stira.reconstruction as reconstruction
    import stira.data as data

    # Set up listmode-to-sinograms converter object
    converter = ListmodeToSinograms(listmode_file, sinograms_file_prefix, raw_data_template, scanning_time_interval, reconstruction_engine, acquisition_data_storage_scheme)

    # Set time interval and flags for storing delayed coincidences
    converter.set_time_interval(scanning_time_interval)
    converter.set_store_delayed_coincidences(True)

    # Process data and get access to sinograms
    converter.process_data()
    sinograms = converter.get_sinograms()

    # Estimate randoms from delayed coincidences
    randoms = estimate_randoms(sinograms, scanning_time_interval)

    # Write estimated randoms to file
    np.savetxt(randoms_file, randoms)

    # Copy acquisition data into Python arrays and print out dimensions, total number of delayed coincidences, and estimated randoms
    acquisition_data = converter.get_acquisition_data()
    dimensions = acquisition_data.get_dimensions()
    total_number_of_delayed_coincidences = acquisition_data.get_total_number_of_delayed_coincidences()
    estimated_randoms = acquisition_data.get_estimated_randoms()
    max_values = acquisition_data.get_max_values()
    print(f"Dimensions: {dimensions}")
    print(f"Total number of delayed coincidences: {total_number_of_delayed_coincidences}")
    print(f"Estimated randoms: {estimated_randoms}")
    print(f"Max values: {max_values}")

    # Display a single sinogram if not in non-interactive mode
    if not non_interactive_mode:
        converter.display_sinogram(0)

if __name__ == "__main__":
    main()
```
Note that this code assumes that the necessary modules and files are installed and available in the Python environment. Additionally, the `petmr_data_path` function is not included in the code snippet, but it can be used to set the path to the data files.