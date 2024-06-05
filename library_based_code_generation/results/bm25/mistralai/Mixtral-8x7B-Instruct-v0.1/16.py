 ```python
import argparse
import os
from SIRF.AcquisitionGeometry import create_from_3D_image
from SIRF.AttenuationImage import get_attenuation_image, set_attenuation_image, check_if_less, check_if_equal, set_attenuation_type, get_attenuation_type
from SIRF.LogLikelihood import PoissonLogLikelihoodWithLinearModelForMeanAndProjData
from SIRF.SIRF import ImageData
from SIRF.STIRAcquisitionDataAlgebraFile import TestSTIRAcquisitionDataAlgebraFile, read_from_file
from SIRF.Utils import normalise_zero_and_one, check_if_zero_within_tolerance, check_if_equal_within_tolerance
from SIRF.VoxelsToWorldTransform import create_uniform_image

def create_multiplicative_sinograms(norm_file=None, atten_file=None, template_file=None, ecattemp_file=None, out_file=None, transform=None, transform_type=None, non_interactive=False):
    if not non_interactive:
        parser = argparse.ArgumentParser(description='Generate multiplicative sinograms.')
        parser.add_argument('--norm_file', type=str, help='Path to normalisation data file')
        parser.add_argument('--atten_file', type=str, help='Path to attenuation data file')
        parser.add_argument('--template_file', type=str, help='Path to template sinogram file')
        parser.add_argument('--ecattemp_file', type=str, help='Path to ECAT8 bin normalisation file')
        parser.add_argument('--out_file', type=str, help='Path to output multiplicative sinogram file')
        parser.add_argument('--transform', type=str, help='Transform for attenuation image')
        parser.add_argument('--transform_type', type=str, help='Transform type')
        parser.add_argument('--non_interactive', action='store_true', help='Disable interactive mode')
        args = parser.parse_args()

        norm_file = args.norm_file if args.norm_file else norm_file
        atten_file = args.atten_file if args.atten_file else atten_file
        template_file = args.template_file if args.template_file else template_file
        ecattemp_file = args.ecattemp_file if args.ecattemp_file else ecattemp_file
        out_file = args.out_file if args.out_file else out_file
        transform = args.transform if args.transform else transform
        transform_type = args.transform_type if args.transform_type else transform_type
        non_interactive = args.non_interactive

    if not (norm_file and atten_file and template_file and ecattemp_file and out_file):
        print("Using default files.")
        norm_file = 'default_norm_file'
        atten_file = 'default_atten_file'
        template_file = 'default_template_file'
        ecattemp_file = 'default_ecattemp_file'
        out_file = 'default_out_file'

    if not os.path.isfile(norm_file):
        print(f"File {norm_file} not found. Using default files.")
        norm_file = 'default_norm_file'

    if not os.path.isfile(atten_file):
        print(f"File {atten_file} not found. Using default files.")
        atten_file = 'default_atten_file'

    if not os.path.isfile(template_file):
        print(f"File {template_file} not found. Using default files.")
        template_file = 'default_template_file'

    if not os.path.isfile(ecattemp_file):
        print(f"File {ecattemp_file} not found. Using default files.")
        ecattemp_file = 'default_ecattemp_file'

    acquisition_geometry = create_from_3D_image(template_file)
    acquisition_data = read_from_file(template_file)

    if transform:
        attenuation_image = get_attenuation_image(atten_file)
        attenuation_image = set_attenuation_type(attenuation_image, transform_type)
        attenuation_image = set_attenuation_image(attenuation_image, atten_file, transform)

        if check_if_less(attenuation_image, 0.0):
            attenuation_image = normalise_zero_and_one(attenuation_image)

        acquisition_geometry = create_from_3D_image(attenuation_image)
        acquisition_data = set_mask_from_attenuation_map(acquisition_data, attenuation_image)

    if check_if_zero_within_tolerance(acquisition_data):
        acquisition_data = create_uniform_image(acquisition_geometry, 1.0)

    if norm_file:
        normalisation_data = ImageData(norm_file)
        normalisation_data = normalise_zero_and_one(normalisation_data)
        acquisition_data = PoissonLogLikelihoodWithLinearModelForMeanAndProjData(acquisition_data, normalisation_data)

    if ecattemp_file:
        ecattemp_data = ImageData(ecattemp_file)
        ecattemp_data = normalise_zero_and_one(ecattemp_data)
        acquisition_data = PoissonLogLikelihoodWithLinearModelForMeanAndProjData(acquisition_data, ecattemp_data)

    if not check_if_equal_within_tolerance(acquisition_data, acquisition_data.linear_model.mean):
        acquisition_data = acquisition_data.linear_model.mean

    TestSTIRAcquisitionDataAlgebraFile(acquisition_data, out_file)
    acquisition_data.write_to_file(out_file)

if __name__ == "__main__":
    create_multiplicative_sinograms()
```