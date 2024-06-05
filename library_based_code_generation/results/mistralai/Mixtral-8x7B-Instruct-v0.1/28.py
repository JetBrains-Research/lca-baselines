 ```python
import argparse
import sys
from sirf.IO import Image
from sirf.Filter import Filter
from sirf.ObjectiveFunctions import ObjectiveFunction
from sirf.Prior import Prior
from sirf.Reconstruction import reconstruct

def osmaposl(image, objective_function, prior, filter, n_subsets, n_sub_iterations):
    # Implement OSMAPOSL reconstruction algorithm here
    pass

def main():
    parser = argparse.ArgumentParser(description='OSMAPOSL reconstruction algorithm.')
    parser.add_argument('raw_data_file', help='Raw data file.')
    parser.add_argument('path_to_data_files', help='Path to data files.')
    parser.add_argument('--n_subsets', type=int, default=10, help='Number of subsets (default: 10).')
    parser.add_argument('--n_sub_iterations', type=int, default=5, help='Number of sub-iterations (default: 5).')
    parser.add_argument('--reconstruction_engine', default='OMP', help='Reconstruction engine (default: OMP).')
    parser.add_argument('--no_plots', action='store_true', help='Disable plots.')

    try:
        args = parser.parse_args()

        acquisition_model = sirf.AcquisitionModel(args.raw_data_file)
        acquisition_data = sirf.AcquisitionData(acquisition_model, args.path_to_data_files)
        filter = sirf.Filter(args.reconstruction_engine)
        initial_image_estimate = Image('initial_image_estimate.nii.gz')
        prior = Prior('prior.nii.gz')
        objective_function = ObjectiveFunction(acquisition_data, filter)

        reconstructed_image = osmaposl(initial_image_estimate, objective_function, prior, filter, args.n_subsets, args.n_sub_iterations)

        if not args.no_plots:
            reconstructed_image.show()

    except Exception as e:
        print('Error:', str(e))
        sys.exit(1)

if __name__ == '__main__':
    main()
```