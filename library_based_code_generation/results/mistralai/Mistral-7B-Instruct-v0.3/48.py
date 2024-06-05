 Here is a simplified Python script that follows your instructions. Note that this script assumes the use of the `pyhdf` library for handling HDF5 files, `numpy` for numerical operations, and `argparse` for command-line options. You may need to adjust the code to fit your specific needs.

```python
import argparse
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.stats import maximum_likelihood_estimate

def listmode_to_sinograms(data_path, listmode_file, sinograms_prefix, randoms_file, template_file, time_interval, non_interactive):
    sd = SD(data_path, SDC.READ)
    sinograms_data = sd.select(listmode_file)
    converter = sinograms.ListModeToSinograms(sinograms_data, template_file, time_interval)
    converter.set_store_delayed_coincidences(True)
    converter.convert()

    sinograms = converter.get_sinograms()
    delayed_coincidences = converter.get_delayed_coincidences()
    randoms = maximum_likelihood_estimate(delayed_coincidences, sinograms)

    with open(randoms_file, 'w') as f:
        np.savetxt(f, randoms)

    acquisition_data = converter.get_acquisition_data()
    print(f"Acquisition data dimensions: {acquisition_data.shape}")
    print(f"Total number of delayed coincidences: {delayed_coincidences.size}")
    print(f"Total number of estimated randoms: {randoms.size}")
    print(f"Max value in acquisition data: {np.max(acquisition_data)}")
    print(f"Max value in estimated randoms: {np.max(randoms)}")

    if not non_interactive:
        print("Sinogram:")
        print(sinograms[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to the data files")
    parser.add_argument("listmode_file", help="Listmode file name")
    parser.add_argument("sinograms_prefix", help="Prefix for sinograms file names")
    parser.add_argument("randoms_file", help="File to save estimated randoms")
    parser.add_argument("template_file", help="Template file for listmode-to-sinograms conversion")
    parser.add_argument("time_interval", type=float, help="Scanning time interval")
    parser.add_argument("reconstruction_engine", help="Reconstruction engine for listmode-to-sinograms conversion")
    parser.add_argument("acquisition_data_storage_scheme", help="Acquisition data storage scheme")
    parser.add_argument("--non-interactive", action="store_true", help="Disable displaying a single sinogram")

    args = parser.parse_args()

    try:
        listmode_to_sinograms(args.data_path, args.listmode_file, args.sinograms_prefix, args.randoms_file, args.template_file, args.time_interval, args.non_interactive)
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
```

This script defines a `listmode_to_sinograms` function that performs the conversion and Maximum Likelihood estimation, and a `main` function that handles command-line arguments and calls the `listmode_to_sinograms` function. The script assumes the presence of a `ListModeToSinograms` class, which you would need to implement according to your specific requirements.