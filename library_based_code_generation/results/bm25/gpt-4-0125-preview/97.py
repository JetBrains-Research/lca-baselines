```python
import sys
import matplotlib.pyplot as plt
from sirf.STIR import ScatterEstimator, read_from_file

def main(raw_data_file, randoms_data_file, acf_file, norm_atten_path, norm_file, atten_image_file, output_prefix, non_interactive):
    try:
        # Set up scatter estimator
        scatter_estimator = ScatterEstimator()

        # Load data
        scatter_estimator.set_input(read_from_file(raw_data_file))
        scatter_estimator.set_attenuation_correction_factors(read_from_file(acf_file))
        scatter_estimator.set_randoms(read_from_file(randoms_data_file))

        # Set paths for normalization and attenuation
        scatter_estimator.set_normalisation(read_from_file(f"{norm_atten_path}/{norm_file}"))
        scatter_estimator.set_attenuation_image(read_from_file(f"{norm_atten_path}/{atten_image_file}"))

        # Set output prefix
        scatter_estimator.set_output_prefix(output_prefix)

        # Process scatter estimation
        scatter_estimator.process()

        if not non_interactive:
            # Display scatter estimate
            scatter_estimate = scatter_estimator.get_output()
            scatter_estimate.show(title="Scatter Estimate")

            # Plot sinogram profile
            sinogram = scatter_estimate.as_array()
            plt.figure()
            plt.plot(sinogram[sinogram.shape[0] // 2, sinogram.shape[1] // 2, :])
            plt.title("Sinogram Profile")
            plt.xlabel("Projection")
            plt.ylabel("Counts")
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python script.py <raw_data_file> <randoms_data_file> <acf_file> <norm_atten_path> <norm_file> <atten_image_file> <output_prefix> <non_interactive>")
        sys.exit(1)

    raw_data_file = sys.argv[1]
    randoms_data_file = sys.argv[2]
    acf_file = sys.argv[3]
    norm_atten_path = sys.argv[4]
    norm_file = sys.argv[5]
    atten_image_file = sys.argv[6]
    output_prefix = sys.argv[7]
    non_interactive = sys.argv[8].lower() in ['true', '1', 't', 'y', 'yes']

    main(raw_data_file, randoms_data_file, acf_file, norm_atten_path, norm_file, atten_image_file, output_prefix, non_interactive)
```