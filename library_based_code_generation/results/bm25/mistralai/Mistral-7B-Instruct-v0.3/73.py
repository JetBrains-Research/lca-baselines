 Here is a Python script that follows your instructions, using the SIRF library. Please note that this is a simplified version and assumes that you have the necessary SIRF library installed.

```python
import argparse
import sys
import numpy as np
from sirflab.acquisition import AcquisitionData, Acquisition
from sirflab.conversion import ListmodeToSinograms
from sirflab.algebra import TestSTIRAcquisitionDataAlgebraFile, TestGadgetronAcquisitionDataAlgebra
from sirflab.data_handling import DataHandleVector

def main(args):
    acquisition_data_storage_scheme = args.storage_scheme
    acquisition_data_template = args.template
    listmode_file = args.listmode
    output_prefix = args.output_prefix
    raw_data_template = args.raw_data_template
    scanning_time_interval = args.time_interval
    reconstruction_engine = args.reconstruction_engine
    interactive_mode = not args.no_interactive
    data_handle_vector = DataHandleVector()

    data_handle_vector.set_acquisition_data(
        AcquisitionDataProcessor.preprocess_acquisition_data(
            new_acquisition_data(
                acquisition_data_storage_scheme,
                acquisition_data_template
            )
        )
    )

    converter = ListmodeToSinograms(
        TestSTIRAcquisitionDataAlgebraFile,
        TestGadgetronAcquisitionDataAlgebra,
        data_handle_vector,
        raw_data_template,
        scanning_time_interval,
        reconstruction_engine
    )

    converter.set_input_files([listmode_file])
    converter.set_output_files([f"{output_prefix}_sinograms.h5"])
    converter.set_data_trajectory()
    converter.setup()
    converter.convert()

    sinograms = converter.get_sinograms()
    acquisition_data = from_acquisition_data(data_handle_vector)
    acquisition_data_array = np.array(acquisition_data.data)

    print(f"Acquisition data dimensions: {acquisition_data_array.shape}")

    if not interactive_mode:
        return

    print(f"Acquisition data:\n")
    show_2D_array(acquisition_data_array[0, :, :])

    randoms = estim_randoms(converter)
    randoms_array = np.array(randoms)

    print(f"\nRandoms:\n")
    show_2D_array(randoms_array[0, :, :])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Listmode-to-sinograms conversion")
    parser.add_argument("listmode", help="Path to the listmode file")
    parser.add_argument("--output_prefix", default="sinograms", help="Output file prefix")
    parser.add_argument("--raw_data_template", default=petmr_data_path("petmr_raw_data_template.xml"), help="Raw data template file")
    parser.add_argument("--time_interval", type=float, default=0.001, help="Scanning time interval to convert (s)")
    parser.add_argument("--reconstruction_engine", default="petmr", help="Reconstruction engine (petmr or gadgetron)")
    parser.add_argument("--storage_scheme", default="memory", help="Acquisition data storage scheme (memory or file)")
    parser.add_argument("--template", default=mr_data_path("petmr_template.xml"), help="Acquisition data template file")
    parser.add_argument("--no_interactive", action="store_true", help="Disable interactive mode")

    try:
        args = parser.parse_args()
        main(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
```

This script defines a main function that performs the listmode-to-sinograms conversion using the SIRF library. It accepts command-line options for various parameters and performs the necessary steps to convert the listmode file into sinograms. The script also estimates randoms and optionally shows 2D arrays of the acquisition data and randoms. The main function is wrapped in a try-except block to handle any errors.