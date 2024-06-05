  import sys
import os
import numpy as np
from sirtf.pet import ScatterEstimator

def main():
    # Parse command-line arguments
    if len(sys.argv) != 10:
        print("Usage: python sirtf_scatter_estimation.py <raw_data_file> <randoms_data_file> <attenuation_correction_factors_file> <path_to_normalization_and_attenuation_files> <normalization_file> <attenuation_image_file> <output_prefix> <non-interactive_mode>")
        sys.exit(1)
    raw_data_file = sys.argv[1]
    randoms_data_file = sys.argv[2]
    attenuation_correction_factors_file = sys.argv[3]
    path_to_normalization_and_attenuation_files = sys.argv[4]
    normalization_file = sys.argv[5]
    attenuation_image_file = sys.argv[6]
    output_prefix = sys.argv[7]
    non_interactive_mode = sys.argv[8]

    # Set up the scatter estimator
    scatter_estimator = ScatterEstimator(raw_data_file, randoms_data_file, attenuation_correction_factors_file, path_to_normalization_and_attenuation_files, normalization_file, attenuation_image_file)

    # Perform the scatter estimation
    try:
        scatter_estimate = scatter_estimator.estimate_scatter()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

    # Display the scatter estimate and plot a sinogram profile
    if non_interactive_mode != "1":
        print("Scatter estimate:", scatter_estimate)
        scatter_estimator.plot_sinogram_profile()

    # Save the scatter estimate to a file
    np.save(output_prefix + "_scatter_estimate.npy", scatter_estimate)

if __name__ == "__main__":
    main()