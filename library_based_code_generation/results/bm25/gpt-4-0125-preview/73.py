```python
import sys
import sirf.STIR as pet
from sirf.Utilities import examples_data_path
import numpy as np

def main(data_path, listmode_file, output_prefix, raw_data_template, time_interval, recon_engine, storage_scheme, interactive_mode):
    try:
        # Set acquisition data storage scheme
        pet.AcquisitionData.set_storage_scheme(storage_scheme)

        # Read the acquisition data template
        template = pet.AcquisitionData(raw_data_template)

        # Create a listmode-to-sinograms converter object
        lm2sino = pet.ListmodeToSinograms()

        # Set input, output, and template files
        lm2sino.set_input(listmode_file)
        lm2sino.set_output_prefix(output_prefix)
        lm2sino.set_template(template)

        # Set the time interval
        lm2sino.set_time_interval(*time_interval)

        # Set some flags (assuming flags are available in the API)
        # lm2sino.set_flag("some_flag", True)

        # Set up the converter
        lm2sino.set_up()

        # Perform the conversion
        lm2sino.process()

        # Get access to the sinograms
        sinograms = lm2sino.get_output()

        # Copy the acquisition data into a Python array
        sino_array = sinograms.as_array()

        # Print the acquisition data dimensions
        print("Sinogram dimensions:", sino_array.shape)

        # Optionally show a 2D array of the acquisition data
        if interactive_mode:
            pet.show_2D_array("Sinograms", sino_array[:, sino_array.shape[1] // 2, :])

        # Estimate randoms
        randoms = lm2sino.estimate_randoms()

        # Convert the randoms to an array
        randoms_array = randoms.as_array()

        # Optionally show a 2D array of the randoms
        if interactive_mode:
            pet.show_2D_array("Randoms", randoms_array[:, randoms_array.shape[1] // 2, :])

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    if len(sys.argv) != 10:
        print("Usage: python script.py <data_path> <listmode_file> <output_prefix> <raw_data_template> <start_time> <end_time> <recon_engine> <storage_scheme> <interactive_mode>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    listmode_file = sys.argv[2]
    output_prefix = sys.argv[3]
    raw_data_template = sys.argv[4]
    time_interval = (float(sys.argv[5]), float(sys.argv[6]))
    recon_engine = sys.argv[7]
    storage_scheme = sys.argv[8]
    interactive_mode = sys.argv[9].lower() in ['true', '1', 't', 'y', 'yes']

    main(data_path, listmode_file, output_prefix, raw_data_template, time_interval, recon_engine, storage_scheme, interactive_mode)
```