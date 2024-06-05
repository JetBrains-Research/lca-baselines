  import os
import sys
import numpy as np
from sirf import SIRF
from sirf.io import read_image, write_image
from sirf.util import check_file_exists
from sirf.acquisition_models import AcquisitionModel
from sirf.acquisition_sensitivity_models import AcquisitionSensitivityModel
from sirf.transforms import TransformType

def main():
    # Parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="Path to data files")
    parser.add_argument("--template-sinogram", type=str, help="Path to template sinogram file")
    parser.add_argument("--attenuation-image", type=str, help="Path to attenuation image file")
    parser.add_argument("--ecat8-bin-normalisation", type=str, help="Path to ECAT8 bin normalisation file")
    parser.add_argument("--output-file", type=str, help="Path to output file")
    parser.add_argument("--transform-type", type=str, help="Type of transform for attenuation image (e.g. 'linear', 'log', 'sqrt')")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode")
    args = parser.parse_args()

    # Check if files exist
    if not check_file_exists(args.data_path):
        print("Data path does not exist")
        sys.exit(1)
    if not check_file_exists(args.template_sinogram):
        print("Template sinogram file does not exist")
        sys.exit(1)
    if not check_file_exists(args.attenuation_image):
        print("Attenuation image file does not exist")
        sys.exit(1)
    if not check_file_exists(args.ecat8_bin_normalisation):
        print("ECAT8 bin normalisation file does not exist")
        sys.exit(1)

    # Load data
    template_sinogram = read_image(args.template_sinogram)
    attenuation_image = read_image(args.attenuation_image)
    ecat8_bin_normalisation = read_image(args.ecat8_bin_normalisation)

    # Create acquisition model
    acquisition_model = AcquisitionModel(template_sinogram)

    # Check if norm and attenuation are present
    if args.non_interactive:
        norm = None
        attenuation = None
    else:
        norm = read_image(os.path.join(args.data_path, "norm.nii.gz"))
        attenuation = read_image(os.path.join(args.data_path, "attenuation.nii.gz"))

    # Create acquisition sensitivity model
    acquisition_sensitivity_model = AcquisitionSensitivityModel(acquisition_model, norm, attenuation)

    # Project data if normalisation is added
    if norm is not None:
        acquisition_sensitivity_model.project(norm)

    # Create multiplicative sinogram
    multiplicative_sinogram = acquisition_sensitivity_model.multiplicative_sinogram(attenuation_image)

    # Write output file
    write_image(multiplicative_sinogram, args.output_file)

if __name__ == "__main__":
    main()