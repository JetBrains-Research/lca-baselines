 Here is a Python script that uses the SIRF library for scatter estimation in PET imaging. The script accepts command-line arguments and performs the required tasks.

```python
import sys
import argparse
import numpy as np
import sift

def scatter_estimation(raw_data_file, randoms_data_file, attenuation_correction_factors_file,
                       normalization_path, normalization_file, attenuation_image_file,
                       output_prefix, non_interactive):
    try:
        # Load data
        data = sift.data.load(raw_data_file)
        randoms = sift.data.load(randoms_data_file)
        acf = np.load(attenuation_correction_factors_file)

        # Load normalization and attenuation files
        normalization = sift.data.load(normalization_path + normalization_file)
        attenuation = sift.data.load(attenuation_image_file)

        # Create scatter estimator
        estimator = sift.estimation.ScatterEstimator(data, randoms, acf, normalization, attenuation)

        # Perform scatter estimation
        scatter_estimate = estimator.estimate()

        # Save scatter estimate
        np.save(output_prefix + '_scatter_estimate.npy', scatter_estimate)

        if not non_interactive:
            # Plot sinogram profile
            sift.plot.sinogram_profile(estimator, title='Scatter Estimate')

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_file", help="Path to raw data file")
    parser.add_argument("randoms_data_file", help="Path to randoms data file")
    parser.add_argument("attenuation_correction_factors_file", help="Path to attenuation correction factors file")
    parser.add_argument("normalization_path", help="Path to normalization and attenuation files")
    parser.add_argument("normalization_file", help="Name of normalization file")
    parser.add_argument("attenuation_image_file", help="Path to attenuation image file")
    parser.add_argument("output_prefix", help="Output prefix for scatter estimates")
    parser.add_argument("-n", "--non-interactive", action="store_true", help="Run in non-interactive mode")
    args = parser.parse_args()

    scatter_estimation(args.raw_data_file, args.randoms_data_file, args.attenuation_correction_factors_file,
                       args.normalization_path, args.normalization_file, args.attenuation_image_file,
                       args.output_prefix, args.non_interactive)
```

To run the script, save it as `scatter_estimation.py` and execute it from the command line with the required arguments:

```
python scatter_estimation.py raw_data.dat randoms.dat acf.npy normalization_and_attenuation_files path/to/normalization.dat path/to/attenuation.dat output_prefix -n
```