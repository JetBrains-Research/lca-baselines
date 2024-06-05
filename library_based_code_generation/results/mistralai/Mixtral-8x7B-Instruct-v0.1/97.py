 ```python
import argparse
import os
import sys
from sirf.PET import ScatterEstimator
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Perform scatter estimation in PET imaging using SIRF.')
    parser.add_argument('raw_data_file', help='Path to the raw data file')
    parser.add_argument('randoms_data_file', help='Path to the randoms data file')
    parser.add_argument('attenuation_correction_factors_file', help='Path to the attenuation correction factors file')
    parser.add_argument('--normalization_path', required=True, help='Path to the directory containing normalization and attenuation files')
    parser.add_argument('normalization_file', help='Path to the normalization file within the normalization_path directory')
    parser.add_argument('attenuation_image_file', help='Path to the attenuation image file')
    parser.add_argument('--output_prefix', required=True, help='Prefix for the output scatter estimate files')
    parser.add_argument('--non_interactive', action='store_true', help='Enable non-interactive mode')

    args = parser.parse_args()

    normalization_path = os.path.abspath(args.normalization_path)
    output_prefix = os.path.abspath(args.output_prefix)

    try:
        scatter_estimator = ScatterEstimator()
        scatter_estimator.setInput(raw_data_file=args.raw_data_file,
                                   randoms_data_file=args.randoms_data_file,
                                   attenuation_correction_factors_file=args.attenuation_correction_factors_file,
                                   normalization_file=os.path.join(normalization_path, args.normalization_file),
                                   attenuation_image_file=args.attenuation_image_file)

        scatter_estimator.process()

        scatter_estimate = scatter_estimator.getOutput('scatter_estimate')
        scatter_estimator.write(os.path.join(output_prefix + '_scatter_estimate.h5'))

        if not args.non_interactive:
            plt.imshow(scatter_estimate.as_matrix(), cmap='gray')
            plt.title('Scatter Estimate')
            plt.show()

            scatter_sinogram = scatter_estimate.as_sinogram()
            plt.plot(scatter_sinogram[0, :])
            plt.title('Sinogram Profile of Scatter Estimate')
            plt.show()

    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
```