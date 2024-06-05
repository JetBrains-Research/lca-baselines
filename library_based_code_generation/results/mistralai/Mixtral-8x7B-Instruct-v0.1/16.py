 ```python
import argparse
import os
import sys

import sitkUtils
from pysirf import SIRF, sitk


def create_acquisition_model(sensitivity_model, norm_data=None, attenuation_data=None):
    acquisition_model = sensitivity_model

    if norm_data is not None:
        acquisition_model = sensitivity_model.project(norm_data)

    if attenuation_data is not None:
        acquisition_model = sensitivity_model.project(attenuation_data, attenuation=True)

    return acquisition_model


def main():
    parser = argparse.ArgumentParser(description="Generate multiplicative sinograms.")
    parser.add_argument("data_path", help="Path to data files.")
    parser.add_argument("template_sinogram", help="Template sinogram file.")
    parser.add_argument("attenuation_image", nargs="?", help="Attenuation image file.")
    parser.add_argument("ecat8_bin_norm", help="ECAT8 bin normalisation file.")
    parser.add_argument("output_file", help="Output file.")
    parser.add_argument("--transform", help="Transform for attenuation image.")
    parser.add_argument("--transform-type", default="sitkBSpline", help="Transform type.")
    parser.add_argument("--non-interactive", action="store_true", help="Non-interactive mode.")

    args = parser.parse_args()

    data_path = args.data_path
    template_sinogram = args.template_sinogram
    attenuation_image = args.attenuation_image
    ecat8_bin_norm = args.ecat8_bin_norm
    output_file = args.output_file
    transform_str = args.transform
    transform_type = args.transform_type
    non_interactive = args.non_interactive

    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "data")

    if not os.path.exists(template_sinogram):
        template_sinogram = os.path.join(os.path.dirname(__file__), "data", "template_sinogram.h5")

    if attenuation_image is not None and not os.path.exists(attenuation_image):
        attenuation_image = None

    if not os.path.exists(ecat8_bin_norm):
        ecat8_bin_norm = os.path.join(os.path.dirname(__file__), "data", "ecat8_bin_norm.h5")

    if transform_str is not None and not transform_str.strip():
        transform_str = None

    sirf = SIRF()

    acquisition_model = sirf.create_acquisition_model(template_sinogram, ecat8_bin_norm)

    if attenuation_image is not None:
        attenuation_image_sitk = sitkUtils.read_image(attenuation_image)

        if transform_str is not None:
            transform = sitk.ReadTransform(transform_str)
            attenuation_image_sitk = sitk.Resample(attenuation_image_sitk, attenuation_image_sitk, transform,
                                                   sitk.sitkLinear, 0.0, attenuation_image_sitk.GetPixelIDValue())

        attenuation_data = sirf.create_image_data(attenuation_image_sitk, transform_type)
        acquisition_model = create_acquisition_model(acquisition_model, attenuation_data=attenuation_data)

    if not non_interactive:
        sirf.interactive()

    norm_data = sirf.create_image_data(os.path.join(data_path, "norm.nii.gz"))
    acquisition_model = create_acquisition_model(acquisition_model, norm_data=norm_data)

    sirf.write_sensitivity_map(output_file, acquisition_model)


if __name__ == "__main__":
    main()
```