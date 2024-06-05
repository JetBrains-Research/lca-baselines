```python
import sys
import argparse
import sirf.STIR as pet
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="PET Scatter Estimation using SIRF")
    parser.add_argument("--raw_data", required=True, help="Path to the raw data file")
    parser.add_argument("--randoms_data", required=True, help="Path to the randoms data file")
    parser.add_argument("--acf_file", required=True, help="Path to the attenuation correction factors file")
    parser.add_argument("--norm_path", required=True, help="Path to normalization and attenuation files")
    parser.add_argument("--norm_file", required=True, help="Normalization file")
    parser.add_argument("--att_image", required=True, help="Attenuation image file")
    parser.add_argument("--output_prefix", required=True, help="Output prefix for scatter estimates")
    parser.add_argument("--non_interactive", action='store_true', help="Run in non-interactive mode")

    args = parser.parse_args()

    try:
        # Set file paths
        raw_data_file = args.raw_data
        randoms_data_file = args.randoms_data
        acf_file = args.acf_file
        norm_file = args.norm_file
        att_image_file = args.att_image
        output_prefix = args.output_prefix

        # Load data
        raw_data = pet.AcquisitionData(raw_data_file)
        randoms_data = pet.AcquisitionData(randoms_data_file)
        acfs = pet.AcquisitionData(acf_file)
        norm = pet.AcquisitionData(norm_file)
        att_image = pet.ImageData(att_image_file)

        # Set up scatter estimator
        scatter_estimator = pet.ScatterEstimator()
        scatter_estimator.set_input(raw_data)
        scatter_estimator.set_attenuation_image(att_image)
        scatter_estimator.set_attenuation_correction_factors(acfs)
        scatter_estimator.set_normalisation(norm)
        scatter_estimator.set_randoms(randoms_data)

        # Perform scatter estimation
        scatter_estimator.process()
        scatter_estimate = scatter_estimator.get_output()

        # Save scatter estimate
        scatter_estimate.write(output_prefix + "_scatter_estimate")

        if not args.non_interactive:
            scatter_estimate.show(title="Scatter Estimate")
            # Plot a sinogram profile
            scatter_array = scatter_estimate.as_array()
            plt.figure()
            plt.plot(scatter_array[scatter_array.shape[0]//2, scatter_array.shape[1]//2, :])
            plt.title("Sinogram Profile of Scatter Estimate")
            plt.xlabel("Projection")
            plt.ylabel("Counts")
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
```