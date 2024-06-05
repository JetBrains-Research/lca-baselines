 Here is a Python script that follows your instructions, using the SIRFLIB library. Please note that you need to have SIRFLIB installed to run this code.

```python
import argparse
import numpy as np
from sirflib.reconstruction import ListmodeToSinograms

def main(args):
    storage_scheme = args.storage_scheme
    template_file = args.template_file
    listmode_file = args.listmode_file
    output_prefix = args.output_prefix
    raw_data_template = args.raw_data_template
    time_interval = args.time_interval
    reconstruction_engine = args.reconstruction_engine
    interactive = not args.no_interactive
    converter = ListmodeToSinograms(storage_scheme=storage_scheme,
                                     reconstruction_engine=reconstruction_engine)

    converter.set_input_file(listmode_file)
    converter.set_output_file_prefix(output_prefix)
    converter.set_template_file(template_file)
    converter.set_time_interval(time_interval)
    converter.set_raw_data_template(raw_data_template)
    converter.set_interactive(interactive)
    converter.setup()
    converter.convert()

    sinograms = converter.get_sinograms()
    acquisition_data = np.array(sinograms)
    print(f"Acquisition data dimensions: {acquisition_data.shape}")

    if not args.no_acquisition_data_display:
        print("Acquisition data:")
        print(acquisition_data[:10, :10])

    randoms = converter.estimate_randoms()
    randoms_array = np.array(randoms)
    print(f"Randoms dimensions: {randoms_array.shape}")

    if not args.no_randoms_display:
        print("Randoms:")
        print(randoms_array[:10, :10])

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("listmode_file", help="Path to the listmode file")
    parser.add_argument("--template_file", help="Path to the acquisition data template file")
    parser.add_argument("--output_prefix", help="Prefix for the output files")
    parser.add_argument("--raw_data_template", help="Path to the raw data template file")
    parser.add_argument("--time_interval", type=float, help="Scanning time interval to convert")
    parser.add_argument("--reconstruction_engine", help="Reconstruction engine to use")
    parser.add_argument("--storage_scheme", help="Acquisition data storage scheme")
    parser.add_argument("--no_interactive", action="store_true", help="Disable interactive mode")
    parser.add_argument("--no_acquisition_data_display", action="store_true", help="Do not display acquisition data")
    parser.add_argument("--no_randoms_display", action="store_true", help="Do not display randoms")
    args = parser.parse_args()
    main(args)
except Exception as e:
    print(f"Error occurred: {e}")
```

This script defines a main function that performs the listmode-to-sinograms conversion using the SIRFLIB library. It accepts command-line options for various parameters and options, and it handles errors using a try-except block. The script also includes options to display the acquisition data and randoms.