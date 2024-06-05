 Here is a simplified Python script that follows your instructions. Please note that this is a basic example and might need adjustments to fit your specific needs. Also, I've assumed that the SIRF library is imported as `sirf`.

```python
import argparse
import sirf
import numpy as np

def create_multiplicative_sinogram(args):
    # Default files
    data_path = 'data_path'
    template_sinogram = 'template_sinogram'
    attenuation_image_file = 'attenuation_image'
    ecat8_bin_normalisation_file = 'ecat8_bin_normalisation'
    output_file = 'output_file'

    # Use provided files if available
    if args.data_path:
        data_path = args.data_path
    if args.template_sinogram:
        template_sinogram = args.template_sinogram
    if args.attenuation_image:
        attenuation_image_file = args.attenuation_image
    if args.ecat8_bin_normalisation:
        ecat8_bin_normalisation_file = args.ecat8_bin_normalisation
    if args.output_file:
        output_file = args.output_file

    # Check if files exist
    if not (os.path.isfile(data_path) and os.path.isfile(template_sinogram) and os.path.isfile(ecat8_bin_normalisation_file) and os.path.isfile(attenuation_image_file)):
        print("Error: Provided files do not exist.")
        return

    # Create acquisition model
    acquisition_model = sirf.TestSTIRAcquisitionDataAlgebraFile(data_path)

    # Check if norm and attenuation are present
    if not acquisition_model.has_norm() and not args.non_interactive:
        print("Error: No normalisation data provided.")
        return
    if not acquisition_model.has_attenuation() and not args.non_interactive:
        print("Error: No attenuation data provided.")
        return

    # Create acquisition sensitivity model
    if args.non_interactive:
        acquisition_sensitivity_model = sirf.create_from_3D_image(acquisition_model.get_attenuation_image())
    else:
        # Handle different types of transformations for the attenuation image
        attenuation_type = args.attenuation_type
        if attenuation_type == 'linear':
            attenuation_image = sirf.create_uniform_image(acquisition_model.get_attenuation_image().shape)
            attenuation_image = sirf.normalise_zero_and_one(attenuation_image)
            attenuation_image *= acquisition_model.get_attenuation_image()
        elif attenuation_type == 'log':
            attenuation_image = np.log(acquisition_model.get_attenuation_image())
        else:
            print("Error: Unsupported attenuation type.")
            return

        # Create acquisition sensitivity model
        acquisition_sensitivity_model = sirf.create_from_3D_image(attenuation_image)

    # Project the data if normalisation is added
    if acquisition_model.has_norm():
        proj_data = sirf.from_acquisition_data(acquisition_model, acquisition_sensitivity_model)
        proj_data = sirf.PoissonLogLikelihoodWithLinearModelForMeanAndProjData(proj_data, acquisition_model.get_norm())
    else:
        proj_data = sirf.from_acquisition_data(acquisition_model, acquisition_sensitivity_model)

    # Set attenuation image and normalisation if available
    if acquisition_model.has_attenuation():
        acquisition_sensitivity_model.set_attenuation_image(acquisition_model.get_attenuation_image())
    if acquisition_model.has_norm():
        acquisition_sensitivity_model.set_norm(acquisition_model.get_norm())

    # Write the multiplicative sinogram to the specified output file
    sirf.write_to_file(acquisition_sensitivity_model.get_multiplicative_sinogram(), output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to data files')
    parser.add_argument('--template_sinogram', type=str, help='Path to template sinogram')
    parser.add_argument('--attenuation_image', type=str, help='Path to attenuation image file')
    parser.add_argument('--ecat8_bin_normalisation', type=str, help='Path to ECAT8 bin normalisation file')
    parser.add_argument('--output_file', type=str, help='Output file')
    parser.add_argument('--attenuation_type', type=str, choices=['linear', 'log'], help='Attenuation type (linear or log)')
    parser.add_argument('--non_interactive', action='store_true', help='Non-interactive mode')
    args = parser.parse_args()
    create_multiplicative_sinogram(args)
```

This script uses the argparse library to handle command-line options. It checks if the provided files exist, creates an acquisition model, checks if norm and attenuation are present, creates an acquisition sensitivity model, projects the data if normalisation is added, and writes the multiplicative sinogram to the specified output file. It also handles different types of transformations for the attenuation image.