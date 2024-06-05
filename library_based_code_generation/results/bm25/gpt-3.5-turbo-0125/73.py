import sys
from sirf.STIR import *

def main():
    data_path = ''
    listmode_file = ''
    output_prefix = ''
    raw_data_template = ''
    scanning_time_interval = 0
    reconstruction_engine = ''
    acquisition_data_storage_scheme = ''
    interactive_mode = True

    # Process command-line options

    # Set acquisition data storage scheme
    storage_scheme = AcquisitionData.get_storage_scheme(acquisition_data_storage_scheme)

    # Read acquisition data template
    acq_data = AcquisitionData(raw_data_template)

    # Create listmode-to-sinograms converter object
    lm2s = ListmodeToSinograms()

    # Set input, output, and template files
    lm2s.set_input(listmode_file)
    lm2s.set_output_prefix(output_prefix)
    lm2s.set_template(acq_data)

    # Set time interval
    lm2s.set_time_interval(scanning_time_interval)

    # Set flags
    lm2s.set_up(acq_data, storage_scheme)

    # Perform conversion
    lm2s.process()

    # Get access to sinograms
    sinograms = lm2s.get_output()

    # Copy acquisition data into Python array
    acq_array = sinograms.as_array()

    # Print acquisition data dimensions
    print(acq_array.shape)

    # Optionally show 2D array of acquisition data
    if not interactive_mode:
        show_2D_array(acq_array)

    # Estimate randoms
    randoms = lm2s.estimate_randoms()

    # Convert randoms to array
    randoms_array = randoms.as_array()

    # Optionally show 2D array of randoms
    if not interactive_mode:
        show_2D_array(randoms_array)

try:
    main()
except error as e:
    print('Error:', e)