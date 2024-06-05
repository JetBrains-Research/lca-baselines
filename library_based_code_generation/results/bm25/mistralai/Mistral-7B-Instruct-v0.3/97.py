 Here is a Python script that uses the SIRF library to perform scatter estimation in PET imaging. The script accepts command-line arguments and performs scatter estimation based on the provided options.

```python
import argparse
import sys
import numpy as np
import sirdm

def main():
    parser = argparse.ArgumentParser(description="Scatter estimation in PET imaging using SIRF library")
    parser.add_argument("data_file", help="Path to raw data file")
    parser.add_argument("randoms_file", help="Path to randoms data file")
    parser.add_argument("ac_factors_file", help="Path to attenuation correction factors file")
    parser.add_argument("norm_attenuation_files", nargs='+', help="Path to normalization and attenuation files")
    parser.add_argument("norm_file", help="Path to normalization file")
    parser.add_argument("attenuation_image_file", help="Path to attenuation image file")
    parser.add_argument("output_prefix", help="Output prefix for scatter estimates")
    parser.add_argument("-n", "--non-interactive", action="store_true", help="Run in non-interactive mode")

    args = parser.parse_args()

    # Set up scatter estimator
    estimator = sirdm.ScatterEstimator()

    # Set collimator and detector files
    estimator.set_collimator_file(args.data_file)
    estimator.set_detector_file(args.data_file)

    # Set attenuation correction factors
    estimator.set_attenuation_correction_factors(args.ac_factors_file)

    # Set normalization and attenuation files
    for file in args.norm_attenuation_files:
        estimator.read_from_file(file)

    # Set normalization file
    estimator.set_normalization_file(args.norm_file)

    # Set attenuation image
    estimator.set_attenuation_image(args.attenuation_image_file)

    # Set output prefix
    estimator.set_output_prefix(args.output_prefix)

    # Perform scatter estimation
    try:
        estimator.process()
    except Exception as e:
        print(f"Error during scatter estimation: {e}")
        sys.exit(1)

    # Normalize zero and one
    estimator.normalise_zero_and_one()

    # Get attenuation image
    attenuation_image = estimator.get_attenuation_image()

    # Estimate randoms
    randoms = estimator.estimate_randoms()

    # Print scatter estimate and plot sinogram profile if not in non-interactive mode
    if not args.non_interactive:
        print(f"Scatter estimate:\n{estimator.get_scatter_estimate()}")
        sirdm.plot_sinogram_profile(estimator.get_sinogram(), randoms, attenuation_image)

if __name__ == "__main__":
    main()
```

To run the script, save it as `scatter_estimation.py` and execute it using the following command:

```bash
python scatter_estimation.py data.sin raw_randoms.sin ac_factors.txt norm_attenuation1.txt norm_attenuation2.txt norm.txt attenuation.mhd output_prefix
```

Replace the file paths and arguments with your own data. The non-interactive mode can be enabled by adding the `-n` or `--non-interactive` flag.